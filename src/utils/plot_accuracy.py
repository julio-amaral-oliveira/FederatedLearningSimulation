# plot_accuracy.py - Módulo unificado para gráficos de acurácia
#
# Pode ser usado tanto pelo modo síncrono quanto assíncrono.
# Uso standalone (da raiz do projeto):
#   python -m src.utils.plot_accuracy --output-dir <dir> [--non-iid] [--alpha 0.1]
#
# Gera visualização:
#   1) Curvas suavizadas sobrepostas (EMA - Média Móvel Exponencial)
#      com TEMPO no eixo X (minutos), refletindo o custo real da simulação.

import argparse
import glob
import json
import os
import re

import numpy as np

try:
    from utils.ema import exponential_moving_average
except ImportError:
    from src.utils.ema import exponential_moving_average


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {"25": "#1f77b4", "50": "#ff7f0e", "75": "#2ca02c", "include_no_timeout": "#d62728"}


def normalize_label(label):
    """Normaliza labels como '25.0' para '25'."""
    try:
        return str(int(float(label)))
    except ValueError:
        return label


def get_color(label):
    return COLORS.get(normalize_label(label), "#9467bd")


def sort_key(label):
    """Ordena labels numéricos primeiro, 'include_no_timeout' por último."""
    try:
        return float(label)
    except ValueError:
        return float("inf")


def get_display_label(label):
    """Retorna o texto de exibição para a legenda do gráfico."""
    normalized = normalize_label(label)
    if normalized == "include_no_timeout":
        return "Sem timeout (100%)"
    try:
        float(normalized)
        return f"p{normalized}"
    except ValueError:
        return normalized


def _infer_dataset(output_dir):
    """Infere o nome amigável do dataset a partir do diretório de saída."""
    base = os.path.basename(os.path.normpath(output_dir))
    mapping = {
        "cifar-10": "CIFAR-10",
        "mnist": "MNIST",
        "fashion-mnist": "Fashion-MNIST",
        "gtsrb": "GTSRB",
    }
    return mapping.get(base, base)


# ---------------------------------------------------------------------------
# Carregamento de dados (com auto-descoberta)
# ---------------------------------------------------------------------------


def load_data(output_dir, is_non_iid, filename=None):
    """Carrega dados de acurácia do JSON.

    Se ``filename`` não for fornecido, tenta o padrão
    ``accuracy_data_{iid|non_iid}.json``.  Caso não exista, procura por
    ``accuracy_data_*.json`` no diretório.  Se houver apenas um match,
    usa-o automaticamente; se houver vários, levanta um erro amigável
    listando as opções.

    Parâmetros:
        output_dir : diretório onde estão os arquivos JSON.
        is_non_iid : se True, carrega dados non-IID.
        filename   : nome do arquivo JSON dentro do diretório. Se None, usa
                     heurística de auto-descoberta.

    Retorna:
        (dados, nome_do_arquivo_usado)
    """
    if filename:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Arquivo especificado não encontrado:\n  {filepath}"
            )
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f), filename

    # 1) tentar o nome padrão
    default_name = (
        "accuracy_data_non_iid.json" if is_non_iid else "accuracy_data_iid.json"
    )
    filepath = os.path.join(output_dir, default_name)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f), default_name

    # 2) auto-descoberta por glob
    pattern = os.path.join(output_dir, "accuracy_data_*.json")
    matches = sorted(glob.glob(pattern))

    if len(matches) == 1:
        used_name = os.path.basename(matches[0])
        with open(matches[0], "r", encoding="utf-8") as f:
            return json.load(f), used_name

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Múltiplos arquivos encontrados em '{output_dir}':\n"
            + "\n".join(f"  - {os.path.basename(m)}" for m in matches)
            + "\n\nUse --filename para escolher um explicitamente."
        )

    raise FileNotFoundError(
        f"Nenhum arquivo accuracy_data_*.json encontrado em '{output_dir}'."
    )


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


def plot_smoothed_overlay(
    data,
    output_dir,
    is_non_iid,
    alpha=0.1,
    dataset_name=None,
    mode=None,
    json_filename=None,
):
    """Gráfico: Todas as curvas suavizadas sobrepostas, com TEMPO no eixo X.

    O eixo X representa o tempo decorrido da simulação (em minutos),
    extraído do campo ``"time"`` de cada avaliação.  Isso reflete
    corretamente o custo temporal de diferentes configurações de timeout.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, entries in sorted(data.items(), key=lambda x: sort_key(x[0])):
        points = sorted(entries, key=lambda x: x["time"])
        times = np.array([p["time"] for p in points]) / 60.0  # segundos → minutos
        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha)

        color = get_color(label)
        display = get_display_label(label)
        ax.plot(
            times,
            smoothed,
            label=f"{display} (EMA, α={alpha})",
            linewidth=2,
            color=color,
        )
        ax.plot(times, acc, alpha=0.35, linewidth=0.5, color=color)

    ax.set_xlabel("Tempo (min)", fontsize=12)
    ax.set_ylabel("Acurácia do modelo", fontsize=12)

    # Título dinâmico
    dataset = dataset_name or _infer_dataset(output_dir)
    dist_str = "Non-IID" if is_non_iid else "IID"
    mode_str = f" — {mode}" if mode else ""
    ax.set_title(
        f"{dataset} ({dist_str}){mode_str} — Acurácia × Tempo",
        fontsize=13,
    )

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Nome do arquivo: baseado no nome do JSON de entrada
    if json_filename:
        png_name = os.path.splitext(os.path.basename(json_filename))[0] + ".png"
    else:
        suffix = "non_iid" if is_non_iid else "iid"
        png_name = f"accuracy_{suffix}.png"
    path = os.path.join(output_dir, png_name)
    fig.savefig(path, dpi=150)
    print(f"Gráfico salvo: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Função de conveniência
# ---------------------------------------------------------------------------


def generate_all_plots(
    output_dir,
    is_non_iid,
    alpha=0.1,
    filename=None,
    dataset_name=None,
    mode=None,
):
    """Carrega os dados e gera o gráfico de curvas suavizadas.

    Parâmetros:
        output_dir   : diretório com os JSONs e onde salvar os PNGs.
        is_non_iid   : se True, usa dados non-IID.
        alpha        : fator de suavização EMA (default: 0.1).
        filename     : nome do arquivo JSON dentro do diretório.
                       Se None, usa heurística de auto-descoberta.
        dataset_name : nome amigável do dataset (ex: "CIFAR-10").
                       Se None, infere do nome do diretório.
        mode         : rótulo do modo (ex: "Síncrono", "Assíncrono").
    """
    data, used_filename = load_data(output_dir, is_non_iid, filename=filename)

    print("=== Curvas suavizadas (Acurácia × Tempo) ===")
    plot_smoothed_overlay(
        data,
        output_dir,
        is_non_iid,
        alpha=alpha,
        dataset_name=dataset_name,
        mode=mode,
        json_filename=used_filename,
    )


# ---------------------------------------------------------------------------
# Execução standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gráficos de acurácia (módulo unificado) — eixo X em tempo real"
    )
    parser.add_argument("--non-iid", action="store_true", help="Usar dados non-IID")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Fator de suavização EMA (default: 0.1). Menor = mais suave.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Diretório com os dados JSON e onde salvar gráficos",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Nome do arquivo JSON dentro do diretório (ex: accuracy_data_non_iid_experimento_teste.json). "
             "Se omitido, tenta auto-descoberta.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Nome amigável do dataset para o título (ex: CIFAR-10). Se omitido, infere do diretório.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Modo do experimento para o título (ex: Síncrono, Assíncrono).",
    )
    args = parser.parse_args()

    generate_all_plots(
        args.output_dir,
        args.non_iid,
        alpha=args.alpha,
        filename=args.filename,
        dataset_name=args.dataset,
        mode=args.mode,
    )
