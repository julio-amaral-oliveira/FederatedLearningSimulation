"""Gráfico do smoke test de drift detection.

Uso (da raiz do projeto):
  python experiments/plot_smoke_drift.py
  python experiments/plot_smoke_drift.py --output-dir output-mnist
  python experiments/plot_smoke_drift.py --output-dir output-mnist --filename smoke_drift.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.ema import exponential_moving_average
except ImportError:
    from src.utils.ema import exponential_moving_average


def main():
    parser = argparse.ArgumentParser(description="Gráfico do smoke test de drift")
    parser.add_argument(
        "--output-dir",
        default="output-mnist",
        help="Diretório com o JSON de saída (default: output-mnist)",
    )
    parser.add_argument(
        "--filename",
        default="smoke_drift.json",
        help="Nome do arquivo JSON (default: smoke_drift.json)",
    )
    args = parser.parse_args()

    filepath = os.path.join(args.output_dir, args.filename)
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        print("Rode primeiro: python experiments/smoke_drift.py --dataset mnist")
        sys.exit(1)

    with open(filepath) as f:
        raw = json.load(f)

    # A primeira chave que não é _meta é a chave dos dados
    data_key = next(k for k in raw if k != "_meta")
    entries = raw[data_key]
    meta = raw.get("_meta", {})

    if not entries:
        print("Nenhum entry encontrado no JSON.")
        sys.exit(1)

    # Ordena por tempo
    entries = sorted(entries, key=lambda e: e["time"])
    times = np.array([e["time"] for e in entries]) / 60.0  # s -> min
    acc = np.array([e["accuracy"] for e in entries])
    smoothed = exponential_moving_average(acc, alpha=0.15)
    drift_flags = np.array([e.get("drift_event", False) for e in entries])
    score = np.array([e.get("mean_score", 0.0) for e in entries])

    # Eventos de drift reais (fronteiras de fase)
    drift_times = meta.get("drift_events", [])
    retrain_times = meta.get("retrain_decisions", [])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Painel 1: Acurácia ---
    ax1.plot(times, acc, alpha=0.3, color="blue", linewidth=0.8, label="Acurácia (bruta)")
    ax1.plot(times, smoothed, color="blue", linewidth=2, label=f"Acurácia (suavizada)")

    # Marca onde o detector flagou drift
    drift_idx = np.where(drift_flags)[0]
    if len(drift_idx) > 0:
        ax1.scatter(
            times[drift_idx],
            smoothed[drift_idx],
            color="red",
            s=40,
            zorder=5,
            marker="v",
            label="Drift detectado",
        )

    # Linhas verticais: drifts reais (fronteiras de fase)
    for t in meta.get("drift_events", []):
        time_min = t.get("time", 0) / 60.0
        ax1.axvline(x=time_min, color="gray", linestyle="--", alpha=0.6)
    # primeira e única linha de fronteira visível
    # usando o schedule.drift_times

    # tenta inferir T_drift da chave do JSON
    import re

    m = re.search(r"T_drift_(\d+\.?\d*)", data_key)
    if m:
        T_drift = float(m.group(1)) / 60.0
        for phase in range(1, 10):
            t = phase * T_drift
            ax1.axvline(x=t, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax1.text(t, ax1.get_ylim()[1] * 0.95, f"Drift\nfase {phase}", fontsize=8,
                     ha="center", va="top", color="gray")

    ax1.set_ylabel("Acurácia", fontsize=12)
    ax1.set_title("Drift Detection — Smoke Test", fontsize=14)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # --- Painel 2: Score ---
    ax2.plot(times, score, color="orange", linewidth=1.5, label="Score médio do detector")
    if len(drift_idx) > 0:
        ax2.scatter(
            times[drift_idx],
            score[drift_idx],
            color="red",
            s=40,
            zorder=5,
            marker="v",
            label="Drift detectado",
        )
    for phase in range(1, 10):
        t = phase * T_drift
        ax2.axvline(x=t, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax2.set_xlabel("Tempo (min)", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = os.path.join(args.output_dir, "smoke_drift_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"Gráfico salvo: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
