"""Streamlit UI para visualizar resultados do smoke_drift.

Uso:
    streamlit run experiments/smoke_drift_ui.py
"""

import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

_BASE = os.path.dirname(__file__)
_SRC = os.path.join(_BASE, "..", "src")
sys.path.insert(0, _SRC)
sys.path.insert(0, _BASE)

import numpy as np

try:
    from utils.ema import exponential_moving_average
except ImportError:
    from src.utils.ema import exponential_moving_average


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_json_files(output_dir: str) -> list[tuple[str, str]]:
    """Retorna [(caminho_completo, nome_arquivo)] de smoke_drift*.json."""
    path = Path(output_dir)
    if not path.exists():
        return []
    files = sorted(path.glob("smoke_drift*.json"))
    return [(str(f), f.name) for f in files]


def _parse_drift_key(data: dict) -> tuple[str, list[dict], dict]:
    """Extrai (key, entries, meta) do JSON do smoke_drift."""
    meta = data.get("_meta", {})
    key = next((k for k in data if k != "_meta"), None)
    if key is None:
        return "unknown", [], meta
    return key, data[key], meta


def _infer_T_drift(key: str) -> float | None:
    m = re.search(r"T_drift_(\d+\.?\d*)", key)
    return float(m.group(1)) if m else None


def _time_to_acc(points_sorted, target):
    for e in points_sorted:
        if e["accuracy"] >= target:
            return e["time"]
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_figure(key, entries, meta, alpha, include_ema, drift_times, target_accs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        return None

    points = sorted(entries, key=lambda e: e["time"])
    times = np.array([e["time"] for e in points]) / 60.0
    acc = np.array([e["accuracy"] for e in points])
    smoothed = exponential_moving_average(acc, alpha)
    drift_flags = np.array([e.get("drift_event", False) for e in points])
    score = np.array([e.get("mean_score", 0.0) for e in points])
    loss = np.array([e.get("loss", 0.0) for e in points])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ---- Painel 1: Acurácia ----
    ax1.plot(times, acc, alpha=0.3, color="blue", linewidth=0.8, label="Bruta")
    if include_ema:
        ax1.plot(times, smoothed, color="blue", linewidth=2,
                 label=f"Suavizada (EMA a={alpha})")

    drift_idx = np.where(drift_flags)[0]
    if len(drift_idx) > 0:
        ax1.scatter(times[drift_idx], acc[drift_idx] if not include_ema else smoothed[drift_idx],
                    color="red", s=45, zorder=5, marker="v", label="Drift detectado")

    for dt_min in drift_times:
        ax1.axvline(x=dt_min, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(dt_min, ax1.get_ylim()[1] * 0.97, f"drift\nt={dt_min*60:.0f}s",
                 ha="center", va="top", fontsize=7, color="gray")

    if target_accs:
        for t in target_accs:
            ax1.axhline(y=t, color="black", linestyle=":", alpha=0.3, linewidth=0.8)

    ax1.set_ylabel("Acurácia", fontsize=11)
    ax1.set_title(f"Smoke Drift — {key}", fontsize=13)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # ---- Painel 2: Score do detector ----
    ax2.plot(times, score, color="orange", linewidth=1.5, label="Score médio do detector")
    if len(drift_idx) > 0:
        ax2.scatter(times[drift_idx], score[drift_idx],
                    color="red", s=45, zorder=5, marker="v", label="Drift detectado")
    for dt_min in drift_times:
        ax2.axvline(x=dt_min, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ---- Painel 3: Loss ----
    ax3.plot(times, loss, color="green", linewidth=1.2, label="Loss")
    if len(drift_idx) > 0:
        ax3.scatter(times[drift_idx], loss[drift_idx],
                    color="red", s=45, zorder=5, marker="v")
    for dt_min in drift_times:
        ax3.axvline(x=dt_min, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax3.set_xlabel("Tempo virtual (min)", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Smoke Drift UI", layout="wide")
    st.title("🧪 Smoke Drift")
    st.markdown("Visualização dos resultados do `experiments/smoke_drift.py`.")

    # ---- Sidebar ----
    st.sidebar.header("Dados")

    dataset_options = {
        "MNIST": "output-mnist",
        "CIFAR-10": "output-cifar-10",
        "Fashion-MNIST": "output-fashion-mnist",
        "GTSRB": "output-gtsrb",
    }
    selected_dataset = st.sidebar.selectbox(
        "Dataset", list(dataset_options.keys()), index=0
    )
    output_dir = dataset_options[selected_dataset]

    json_files = _discover_json_files(output_dir)
    if not json_files:
        st.sidebar.warning(
            f"Nenhum arquivo smoke_drift*.json encontrado em `{output_dir}`.\n\n"
            "Rode primeiro:\n"
            "```bash\n"
            f"python experiments/smoke_drift.py --dataset {selected_dataset.lower()}\n"
            "```"
        )
        st.stop()

    filepath, filename = json_files[0] if len(json_files) == 1 else json_files[0]

    if len(json_files) > 1:
        selected_file = st.sidebar.selectbox(
            "Arquivo JSON",
            options=[f for _, f in json_files],
            format_func=lambda x: x,
            index=0,
        )
        filepath = os.path.join(output_dir, selected_file)
    else:
        st.sidebar.info(f"Arquivo: `{filename}`")

    with open(filepath) as f:
        raw = json.load(f)
    key, entries, meta = _parse_drift_key(raw)

    if not entries:
        st.error("Nenhum dado encontrado no JSON.")
        st.stop()

    st.sidebar.header("Config. do gráfico")
    alpha = st.sidebar.slider("Suavização EMA (alpha)", 0.01, 0.5, 0.1, 0.01)
    include_ema = st.sidebar.checkbox("Mostrar curva suavizada", value=True)

    target_input = st.sidebar.text_input(
        "Alvos de acurácia (vírgula)",
        value="0.50, 0.60, 0.70",
        help="Linhas horizontais no gráfico",
    )

    # ---- Métricas extra do experimento ----
    drift_events = meta.get("drift_events", [])
    retrain_decisions = meta.get("retrain_decisions", [])

    # ---- Main area ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fases detectadas", len(set(e.get("phase", 0) for e in entries)))
    with col2:
        st.metric("Eventos de drift", len(drift_events))
    with col3:
        st.metric("Decisões de retreino", len(retrain_decisions))
    with col4:
        unique_clients = len(set(e.get("client_id", "?") for e in drift_events))
        st.metric("Clientes c/ drift", unique_clients if drift_events else 0)

    # ---- Drift times ----
    T_drift = _infer_T_drift(key)
    drift_times_min = []
    if T_drift:
        max_phase = max(e.get("phase", 1) for e in entries) if entries else 1
        drift_times_min = [(i * T_drift) / 60.0 for i in range(1, max_phase)]

    # ---- Plot ----
    try:
        targets = [float(v.strip()) for v in target_input.split(",") if v.strip()]
    except ValueError:
        targets = []

    fig = _build_figure(key, entries, meta, alpha, include_ema, drift_times_min, targets)
    if fig:
        st.subheader("Curvas de detecção de drift")
        st.pyplot(fig)
    else:
        st.warning("Não foi possível gerar o gráfico.")

    # ---- Eventos de drift ----
    st.subheader("Eventos de drift detectados")
    if drift_events:
        st.dataframe(drift_events, use_container_width=True)
    else:
        st.info("Nenhum evento de drift registrado.")

    # ---- Decisões de retreino ----
    if retrain_decisions:
        st.subheader("Decisões de re-treino")
        st.dataframe(retrain_decisions, use_container_width=True)

    # ---- Detector info ----
    st.subheader("Configuração do detector")
    detector_info = {
        "detector_kind": meta.get("detector_kind", "?"),
        "trigger_threshold": meta.get("trigger_threshold", "?"),
    }
    detector_info["alpha (ADWIN delta)"] = meta.get("alpha", "?")
    detector_info["T (MC Dropout passes)"] = meta.get("T", "?")
    st.json(detector_info)

    # ---- Comparação: carregar outro JSON ----
    st.divider()
    st.subheader("Comparar com outro experimento")

    other_files = [f for f in json_files if f[1] != filename]
    if other_files:
        other_path = st.selectbox(
            "Segundo arquivo",
            options=[f[1] for f in other_files],
            key="compare_file",
        )
        if st.button("Comparar"):
            with open(os.path.join(output_dir, other_path)) as f2:
                raw2 = json.load(f2)
            key2, entries2, meta2 = _parse_drift_key(raw2)

            if entries2:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                points1 = sorted(entries, key=lambda e: e["time"])
                points2 = sorted(entries2, key=lambda e: e["time"])

                times1 = np.array([e["time"] for e in points1]) / 60.0
                acc1 = np.array([e["accuracy"] for e in points1])
                times2 = np.array([e["time"] for e in points2]) / 60.0
                acc2 = np.array([e["accuracy"] for e in points2])

                figc, axc = plt.subplots(figsize=(12, 5))
                axc.plot(times1, exponential_moving_average(acc1, alpha),
                         linewidth=2, label=f"Run 1: {key}")
                axc.plot(times2, exponential_moving_average(acc2, alpha),
                         linewidth=2, label=f"Run 2: {key2}")
                for dt_min in drift_times_min:
                    axc.axvline(x=dt_min, color="gray", linestyle="--", alpha=0.4)

                axc.set_xlabel("Tempo (min)")
                axc.set_ylabel("Acurácia (suavizada)")
                axc.set_title("Comparação entre execuções")
                axc.legend()
                axc.grid(True, alpha=0.3)
                figc.tight_layout()
                st.pyplot(figc)
            else:
                st.warning("Segundo arquivo não contém dados de acurácia.")
    else:
        st.info("Apenas um arquivo smoke_drift encontrado. Rode outro experimento com `--output-prefix` para comparar.")


if __name__ == "__main__":
    main()
