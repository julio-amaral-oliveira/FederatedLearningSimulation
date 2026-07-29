import sys
import subprocess
import os

# --- CONFIGURAÇÕES ---
TARGET_SCRIPT = "smoke_drift.py" 
BASE_OUTPUT_DIR = "output/cifar-10/drift-agent"

# As listas devem ter exatamente o mesmo número de elementos.
# O índice 0 de CORRUPTIONS rodará com o índice 0 de SEVERITIES, e assim por diante.
CORRUPTIONS = ['gaussian_noise', 'frosted_glass_blur', 'motion_blur', 'fog']

SEVERITIES = [
    5, # Severidade para gaussian_noise
    1, # Severidade para frosted_glass_blur
    1,  # Severidade para motion_blur
    1 # Severidade para fog
]

def main():
    # Validação básica de segurança
    if len(CORRUPTIONS) != len(SEVERITIES):
        raise ValueError("Erro: As listas CORRUPTIONS e SEVERITIES precisam ter o mesmo tamanho!")

    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"Erro: O arquivo '{TARGET_SCRIPT}' não foi encontrado neste diretório.")

    print("Iniciando bateria de experimentos...\n" + "="*40)

    # O zip() itera simultaneamente pelas duas listas
    for corruption, severity in zip(CORRUPTIONS, SEVERITIES):
        print(f"\n-> Iniciando: {corruption} (Severidade: {severity})")
        
        # Cria um diretório de saída único para cada execução para não sobrescrever os JSONs
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{corruption}_sev{severity}")
        
        # Monta a chamada de terminal
        command = [
            sys.executable, TARGET_SCRIPT,
            "--corruption", corruption,
            "--severity", str(severity),
            "--output-dir", output_dir
        ]
        
        try:
            # Executa o script. 
            # capture_output=True pega os prints originais, text=True decodifica como string
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            
            print(f"  Experimento finalizado.")
            print(f"  Resultados salvos em: {output_dir}")
            

            print(f"  [Log da execução]:\n{process.stdout}")
            
        except subprocess.CalledProcessError as e:
            print(f"  Erro ao executar o experimento de {corruption}:")
            print(f"Detalhes do erro:\n{e.stderr}")

    print("\n" + "="*40)
    print("Bateria de experimentos concluída com sucesso!")

if __name__ == "__main__":
    main()