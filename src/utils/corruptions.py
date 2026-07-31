import torch

import torchvision.transforms.functional as F
import torch.nn.functional as NN_F

def apply_corruption(
    x: torch.Tensor,
    kind: str,
    severity: int,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Aplica corrupções a um tensor de imagem para fins de teste de robustez.
    Corrupções suportadas: 'gaussian_noise', 'frosted_glass_blur', 'motion_blur', 'fog'.
    """
    
    # 1. Validação de Severidade
    if isinstance(severity, bool) or not isinstance(severity, int) or not (1 <= severity <= 5):
        raise ValueError(f"Severidade inválida: {severity}. Deve ser um número inteiro de 1 a 5.")
        
    # 2. Validação do Tipo de Corrupção
    valid_kinds = ['gaussian_noise', 'frosted_glass_blur', 'motion_blur', 'fog']
    if kind not in valid_kinds:
        raise ValueError(f"Corrupção desconhecida: '{kind}'. Opções suportadas: {valid_kinds}")

    # 3. Cópia e Extração de Dimensões
    out = x.clone()
    device = out.device
    dtype = out.dtype
    channels, height, width = out.shape[-3:]
    is_batched = out.ndim == 4
    
    # 4. Configuração de Seed Isolada
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    # 5. Aplicação da Corrupção
    if kind == 'gaussian_noise':
        std_levels = [0.08, 0.10, 0.14, 0.18, 0.24]
        std = std_levels[severity - 1]
        
        if gen is not None:
            noise = torch.randn(out.shape, generator=gen, device=device, dtype=dtype)
        else:
            noise = torch.randn_like(out)
            
        out = out + noise * std

    elif kind == 'frosted_glass_blur':
        # Simula o efeito de vidro fosco aplicando permutações locais de pixels (shuffle)
        d = 3
        
        # Cria as matrizes de coordenadas espaciais
        yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        
        # Sorteia os deslocamentos para cada pixel individualmente
        if gen is not None:
            dy = torch.randint(-d, d + 1, (height, width), generator=gen, device=device)
            dx = torch.randint(-d, d + 1, (height, width), generator=gen, device=device)
        else:
            dy = torch.randint(-d, d + 1, (height, width), device=device)
            dx = torch.randint(-d, d + 1, (height, width), device=device)
            
        # Garante que as novas coordenadas não ultrapassem os limites da imagem
        yy = torch.clamp(yy + dy, 0, height - 1)
        xx = torch.clamp(xx + dx, 0, width - 1)
        
        # Reposiciona os pixels e mistura progressivamente a mesma distorção
        # aleatória, para que a intensidade aumente com a severidade.
        distorted = out[..., yy, xx]
        
        # Aplica um leve desfoque para emular a dispersão da luz no vidro e suavizar as arestas vivas
        blurred = F.gaussian_blur(distorted, kernel_size=[3, 3], sigma=[1.0, 1.0])
        strength = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
        out = torch.lerp(out, blurred, strength)

    elif kind == 'motion_blur':
        # Cria um kernel de convolução direcional para simular rastros de movimento
        kernel_sizes = [5, 7, 9, 11, 15]
        k = kernel_sizes[severity - 1]
        
        # Cria uma matriz identidade (linha diagonal contínua)
        kernel = torch.eye(k, device=device, dtype=dtype)
        
        # Sorteia a inclinação (diagonal principal ou secundária) para variar o ângulo
        if gen is not None:
            flip = torch.rand(1, generator=gen, device=device).item() > 0.5
        else:
            flip = torch.rand(1, device=device).item() > 0.5
            
        if flip:
            kernel = kernel.flip(1)
            
        # Normaliza o kernel para não aumentar o brilho geral da imagem
        kernel = kernel / kernel.sum() 
        # Modela para o formato [channels, 1, k, k] exigido pela convolução depthwise
        kernel = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
        
        # Aplica o blur via convolução independente por canal
        if not is_batched:
            out = out.unsqueeze(0)
            
        out = NN_F.conv2d(out, kernel, groups=channels, padding=k // 2)
        
        if not is_batched:
            out = out.squeeze(0)

    elif kind == 'fog':
        intensity = 3.0
        wibbledecay = 1.4
        
        # Extrai propriedades do tensor considerando o suporte a batches inserido anteriormente
        is_batched = out.ndim == 4
        channels, height, width = out.shape[-3:]
        batch_size = out.shape[0] if is_batched else 1
        
        # Inicializa o mapa do fractal no formato 4D exigido pelo torch.nn.functional.interpolate
        fog_map = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        
        weight = 1.0
        total_weight = 0.0
        num_octaves = 4  # Quantidade de frequências combinadas
        
        # Geração do Plasma Fractal via Fractional Brownian Motion (fBm)
        for octave in range(num_octaves):
            grid_size = 2 ** (octave + 1)  # Resoluções base crescentes: 2x2, 4x4, 8x8, 16x16
            
            rand_shape = (batch_size, 1, grid_size, grid_size)
            if gen is not None:
                noise = torch.rand(rand_shape, generator=gen, device=device, dtype=dtype)
            else:
                noise = torch.rand(rand_shape, device=device, dtype=dtype)
                
            # O upsampling bilinear contínuo cria o aspecto de nuvem suave
            noise_up = NN_F.interpolate(noise, size=(height, width), mode='bilinear', align_corners=False)
            
            # Acumula a camada aplicando o wibbledecay (controla a rugosidade do fractal)
            fog_map += noise_up * weight
            total_weight += weight
            weight /= wibbledecay
            
        # Normaliza o mapa fractal gerado para o intervalo válido [0, 1]
        fog_map /= total_weight
        
        # Se a entrada era 3D (imagem única), remove a dimensão temporal de batch adicionada
        if not is_batched:
            fog_map = fog_map.squeeze(0)
            
        # Mistura uma única névoa aleatória com intensidade crescente, mantendo
        # a progressão de severidade independente das variações da amostra.
        fogged = (out + intensity * fog_map) / (1.0 + intensity)
        strength = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
        out = torch.lerp(out, fogged, strength)

    # 6. Normalização Final
    out = torch.clamp(out, 0.0, 1.0)
    
    return out


if __name__ == "__main__":
    import os
    import sys
    import random
    import numpy as np
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # --- CONFIGURAÇÕES DA VISUALIZAÇÃO ---
    SEVERITY = 3
    NUM_IMAGES = 5
    SEED = 42

    # --- CONFIGURAÇÃO DE CAMINHOS ---
    _BASE = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.join(_BASE, "..")
    _ROOT = os.path.join(_SRC, "..")
    _EXPERIMENTS = os.path.join(_ROOT, "experiments")
    _DATA = os.path.join(_EXPERIMENTS, "data")

    for _path in (_ROOT, _SRC):
        if _path not in sys.path:
            sys.path.insert(0, _path)

    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Converte um tensor PyTorch (C, H, W) para um array NumPy (H, W, C) para o Matplotlib."""
        img = tensor.detach().cpu().numpy()
        # Rearranja os eixos de (C, H, W) para (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        # Garante que os valores estejam entre 0 e 1 para plotagem correta
        img = np.clip(img, 0.0, 1.0)
        return img

    def visualize_corruption_examples(num_images: int = 5, seed: int = 42):
        print(f"Carregando amostras do CIFAR-10...")
        dataset = torchvision.datasets.CIFAR10(
            root=_DATA, train=False, download=True, transform=transforms.ToTensor()
        )
        
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_images)
        
        corruptions = ['gaussian_noise', 'frosted_glass_blur', 'motion_blur', 'fog']
        severities = [1, 2, 3, 4, 5]
        
        print("Aplicando corrupções e gerando gráficos...")
        
        for kind in corruptions:
            # Cria uma grade: 1 linha para original + 5 linhas para severidades
            nrows = len(severities) + 1
            fig, axes = plt.subplots(nrows=nrows, ncols=num_images, figsize=(num_images * 2.5, nrows * 2.5))
            fig.suptitle(f"Evolução de Severidade: {kind.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
            
            for i, idx in enumerate(indices):
                original_tensor, label = dataset[idx]
                
                # --- Plota a imagem Original (Linha 0) ---
                ax_orig = axes[0, i]
                ax_orig.imshow(tensor_to_image(original_tensor))
                ax_orig.set_title(f"Original\n(Classe {label})")
                ax_orig.set_xticks([])
                ax_orig.set_yticks([])
                
                # Adiciona o rótulo da linha na primeira coluna
                if i == 0:
                    ax_orig.set_ylabel("Original", fontsize=12, fontweight='bold', rotation=0, labelpad=40, ha='right', va='center')
                
                # --- Plota as Corrupções (Linhas 1 a 5) ---
                for s_idx, severity in enumerate(severities):
                    row = s_idx + 1
                    ax_corr = axes[row, i]
                    
                    try:
                        # Aplica a corrupção. Usa o seed base + índice da imagem + severidade para variar o ruído adequadamente
                        corrupted_tensor = apply_corruption(
                            original_tensor, 
                            kind=kind, 
                            severity=severity, 
                            seed=seed + i + severity
                        )
                        ax_corr.imshow(tensor_to_image(corrupted_tensor))
                    except Exception as e:
                        print(f"Erro ao aplicar {kind} (Sev {severity}) na imagem {i}: {e}")
                    
                    ax_corr.set_xticks([])
                    ax_corr.set_yticks([])
                    
                    # Adiciona o rótulo da severidade na primeira coluna de cada linha
                    if i == 0:
                        ax_corr.set_ylabel(f"Severidade {severity}", fontsize=12, fontweight='bold', rotation=0, labelpad=40, ha='right', va='center')

            # Ajusta o layout para não cortar os textos e exibir de forma limpa
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, left=0.20)
            
            output_filename = f"vis_{kind}_all_severities.png"
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"[{kind}] Imagem salva com sucesso em: {output_filename}")
            
            plt.show() 

    visualize_corruption_examples(num_images=5, seed=42)
