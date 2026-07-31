import argparse
import glob
import json
import os
import re

import matplotlib

os.environ["QT_QPA_PLATFORM"] = "offscreen"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.ema import exponential_moving_average
from experiments.registry import temporary_output_path


def parse_experiment_metadata(filename):
    """Extrai metadados do nome do arquivo de resultado."""
    pattern = (
        r"accuracy_data_(?P<distribution>iid|non_iid)_(?P<prefix>[^_]+)_"
        r"A(?P<A>[0-9.]+)_B(?P<B>[0-9.]+)_C(?P<C>[0-9.]+)_D(?P<D>[0-9]+)\.json"
    )
    match = re.match(pattern, filename)
    if not match:
        return None
    return match.groupdict()


PARAM_LABELS = {
    "base_alpha": "Base Alpha (A)",
    "decay": "Decay of Base Alpha (B)",
    "tardiness": "Tardiness Sensitivity (C)",
}

# Valores padrão usados como referência no estudo de ablação
DEFAULTS = {
    "A": "0.8",
    "B": "0.999",
    "C": "0.075",
}


def find_experiment_files(output_dir, distribution, prefix="async"):
    """Encontra e parseia todos os arquivos de experimento."""
    pattern = os.path.join(output_dir, f"accuracy_data_{distribution}_{prefix}*.json")
    files = glob.glob(pattern)
    entries = []
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        meta = parse_experiment_metadata(filename)
        if meta:
            entries.append((filepath, meta))
    return entries


def filter_for_study(entries, vary):
    """Filtra arquivos para um estudo one-at-a-time.

    Mantém apenas os arquivos onde os parâmetros NÃO variados
    estão nos valores padrão.
    """
    # Mapeia o nome do estudo para qual campo do metadata varia
    vary_key = {"base_alpha": "A", "decay": "B", "tardiness": "C"}[vary]
    fixed_keys = [k for k in ["A", "B", "C"] if k != vary_key]

    filtered = []
    for filepath, meta in entries:
        if all(meta[k] == DEFAULTS[k] for k in fixed_keys):
            filtered.append((filepath, meta, float(meta[vary_key])))

    filtered.sort(key=lambda x: x[2])
    return filtered, vary_key


def plot_isolated(output_dir, distribution, percentile, vary, alpha_ema=0.1):
    """Gera gráfico de ablação para um parâmetro variado isoladamente."""
    entries = find_experiment_files(output_dir, distribution)
    filtered, vary_key = filter_for_study(entries, vary)

    if not filtered:
        print(f"[{distribution.upper()}] Nenhum arquivo encontrado para estudo de {vary}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(filtered)))

    for idx, (filepath, meta, value) in enumerate(filtered):
        with open(filepath) as f:
            data = json.load(f)

        if percentile not in data:
            continue

        points = sorted(data[percentile], key=lambda x: x["time"])
        if not points:
            continue

        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha_ema)
        x_axis = np.arange(1, len(acc) + 1)

        ax.plot(x_axis, acc, alpha=0.15, color=colors[idx], linewidth=1)
        ax.plot(
            x_axis,
            smoothed,
            linewidth=2.5,
            color=colors[idx],
            label=f"{PARAM_LABELS[vary]} = {value}",
        )

    fixed_params = {k: DEFAULTS[k] for k in ["A", "B", "C"] if k != vary_key}
    fixed_str = ", ".join(f"{PARAM_LABELS[p]}={v}" for p, v in
                          zip(["base_alpha", "decay", "tardiness"],
                              [DEFAULTS["A"], DEFAULTS["B"], DEFAULTS["C"]])
                          if {"base_alpha": "A", "decay": "B", "tardiness": "C"}[p] != vary_key)

    ax.set_xlabel("Número de atualizações", fontsize=12)
    ax.set_ylabel("Acurácia", fontsize=12)
    ax.set_title(
        f"Ablação: {PARAM_LABELS[vary]} ({distribution.upper()}, p{percentile})",
        fontsize=13,
    )
    ax.text(
        0.02, 0.02, f"Fixos: {fixed_str}",
        transform=ax.transAxes, fontsize=9, alpha=0.7, va="bottom",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(
        output_dir, f"ablation_{vary}_{distribution}_p{percentile}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo: {out_path}")
    plt.close(fig)


def plot_grid(
    output_dir, distribution, prefix, percentile, alpha_ema=0.1, show_raw=False
):
    """Gráfico facetado (grid completo) — modo legado para grid search."""
    from matplotlib.lines import Line2D

    pattern = os.path.join(output_dir, f"accuracy_data_{distribution}_{prefix}*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"[{distribution.upper()}:{prefix.upper()}] Nenhum arquivo encontrado: {pattern}")
        return

    file_entries = []
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        metadata = parse_experiment_metadata(filename)
        if metadata:
            file_entries.append((filepath, filename, metadata))

    if not file_entries:
        return

    a_values = sorted({e[2]["A"] for e in file_entries}, key=float)
    b_values = sorted({e[2]["B"] for e in file_entries}, key=float)
    c_values = sorted({e[2]["C"] for e in file_entries}, key=float)

    decay_palette = plt.cm.rainbow(np.linspace(0, 1, max(3, len(b_values))))
    decay_color_map = {b: decay_palette[i % len(decay_palette)] for i, b in enumerate(b_values)}
    c_ls = ["-", "--", ":", "-."]
    c_mk = ["o", "s", "^", "D", "v", "P", "X"]
    c_ls_map = {c: c_ls[i % len(c_ls)] for i, c in enumerate(c_values)}
    c_mk_map = {c: c_mk[i % len(c_mk)] for i, c in enumerate(c_values)}

    fig, axes = plt.subplots(1, len(a_values), figsize=(5.2 * len(a_values), 5.8), sharex=True, sharey=True)
    if len(a_values) == 1:
        axes = [axes]
    axis_by_a = {a: axes[i] for i, a in enumerate(a_values)}

    for filepath, filename, meta in file_entries:
        ax = axis_by_a[meta["A"]]
        color = decay_color_map[meta["B"]]
        ls = c_ls_map[meta["C"]]
        mk = c_mk_map[meta["C"]]

        with open(filepath) as f:
            data = json.load(f)
        if percentile not in data:
            continue
        points = sorted(data[percentile], key=lambda x: x["time"])
        if not points:
            continue
        acc = np.array([p["accuracy"] for p in points])
        smoothed = exponential_moving_average(acc, alpha_ema)
        x_axis = np.arange(1, len(acc) + 1)

        if show_raw:
            ax.plot(x_axis, acc, alpha=0.1, linewidth=1, color=color, linestyle=ls)
        ax.plot(x_axis, smoothed, linewidth=2.2, color=color, linestyle=ls,
                marker=mk, markevery=max(1, len(x_axis) // 6), markersize=4.5)

    for a in a_values:
        axis_by_a[a].set_title(f"Base alpha = {a}", fontsize=11)
        axis_by_a[a].grid(True, alpha=0.25)
        axis_by_a[a].set_xlabel("Atualizações", fontsize=11)
    axes[0].set_ylabel("Acurácia", fontsize=12)
    fig.suptitle(
        f"Ablação {distribution.upper()}: {prefix.upper()} - p{percentile}",
        fontsize=14, y=0.98,
    )

    decay_handles = [
        Line2D([0], [0], color=decay_color_map[b], lw=2.5, label=f"Decay = {b}")
        for b in b_values
    ]
    tard_handles = [
        Line2D([0], [0], color="black", lw=2, marker=c_mk_map[c], linestyle=c_ls_map[c],
               markersize=6, label=f"Tardiness = {c}")
        for c in c_values
    ]
    fig.legend(handles=decay_handles + tard_handles, fontsize=9,
               loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=True)
    fig.subplots_adjust(right=0.81, wspace=0.10)

    out_path = os.path.join(output_dir, f"ablation_comparison_{distribution}_{prefix}_p{percentile}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo: {out_path}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gera gráficos de ablação")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(temporary_output_path("e02-ablation", "cifar-10")),
    )
    parser.add_argument("--distribution", type=str, default="iid", help="iid, non_iid, ou all")
    parser.add_argument("--percentile", type=str, default="50")
    parser.add_argument("--vary", type=str, default="all",
                        help="Parâmetro variado: base_alpha, decay, tardiness, ou all")
    parser.add_argument("--mode", type=str, default="isolated", choices=["isolated", "grid"],
                        help="isolated = one-at-a-time (padrão), grid = grid search completo")
    parser.add_argument("--prefix", type=str, default="async", help="Prefixo (só para modo grid)")
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args(argv)

    distributions = ["iid", "non_iid"] if args.distribution == "all" else [args.distribution]

    if args.mode == "isolated":
        params = ["base_alpha", "decay", "tardiness"] if args.vary == "all" else [args.vary]
        for dist in distributions:
            for vary in params:
                plot_isolated(args.output_dir, dist, args.percentile, vary)
    else:
        for dist in distributions:
            plot_grid(args.output_dir, dist, args.prefix, args.percentile,
                      show_raw=args.show_raw)


if __name__ == "__main__":
    main()
