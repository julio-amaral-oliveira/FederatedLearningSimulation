"""Small Streamlit UI for temporal drift plots.

Run:
    streamlit run experiments/drift_ui.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

try:
    from experiments.plot_drift import (
        _parse_float_list,
        plot_compare_mode,
        plot_single_mode,
    )
except ModuleNotFoundError:
    from plot_drift import (
        _parse_float_list,
        plot_compare_mode,
        plot_single_mode,
    )


def main():
    st.set_page_config(page_title="Temporal Drift Plots", layout="wide")
    st.title("Temporal Drift — Visualizacao")

    st.write(
        "Gera graficos de acuracia x tempo virtual com linhas de fase, "
        "acuracia por grupo (A/B), comparacao sync vs async, "
        "e metricas de tempo ate acuracia alvo."
    )

    mode = st.radio(
        "Modo",
        ["single", "compare"],
        format_func=lambda m: "Um cenario" if m == "single" else "Sync vs Async",
        horizontal=True,
    )

    col_left, col_right = st.columns(2)

    if mode == "single":
        with col_left:
            json_file = st.text_input(
                "Arquivo JSON",
                value="output-cifar-10/accuracy_data_iid_T_drift_200_sync.json",
                help="Caminho para o JSON de resultado do temporal_drift.py",
            )
            json_key = st.text_input(
                "Chave (opcional)",
                value="",
                help="Deixe vazio se o JSON tiver uma unica chave.",
            )
        with col_right:
            title = st.text_input(
                "Titulo",
                value="Temporal Drift",
            )
    else:
        with col_left:
            sync_json = st.text_input(
                "Sync JSON",
                value="output-cifar-10/accuracy_data_iid_T_drift_200_sync.json",
            )
            sync_key = st.text_input(
                "Sync chave (opcional)",
                value="",
                key="sync_key",
            )
        with col_right:
            async_json = st.text_input(
                "Async JSON",
                value="output-cifar-10/accuracy_data_iid_T_drift_200_async.json",
            )
            async_key = st.text_input(
                "Async chave (opcional)",
                value="",
                key="async_key",
            )
        title = st.text_input(
            "Titulo do grafico",
            value="Sync vs Async — Temporal Drift",
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        alpha = st.number_input(
            "EMA alpha",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
        )
    with col_b:
        include_ema = st.checkbox("Mostrar EMA", value=True)
    with col_c:
        output_dir = st.text_input(
            "Diretorio de saida",
            value="output",
        )

    output_prefix = st.text_input(
        "Prefixo do arquivo (opcional)",
        value="",
        help="Ex: 'experimento1' gera experimento1_drift_overall_sync.png",
    )

    target_text = st.text_input(
        "Alvos de acuracia",
        value="0.50, 0.60, 0.70",
        help="Valores separados por virgula. Linhas horizontais no grafico + time_to_acc.",
    )

    if st.button("Gerar graficos", type="primary"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            targets = _parse_float_list(target_text)

            if mode == "single":
                key = json_key.strip() if json_key.strip() else None
                plot_single_mode(
                    json_file.strip(),
                    key,
                    output_dir,
                    alpha=alpha,
                    title=title.strip() or None,
                    include_ema=include_ema,
                    target_accuracies=targets if targets else None,
                    output_prefix=output_prefix.strip(),
                )
                st.success("Graficos gerados.")
                prefix = f"{output_prefix.strip()}_" if output_prefix.strip() else ""
                st.subheader("Acurácia × Tempo")
                expected = os.path.join(output_dir, f"{prefix}drift_overall_sync.png")
                if not os.path.exists(expected):
                    import glob
                    candidates = glob.glob(os.path.join(output_dir, f"{prefix}drift_overall_*.png"))
                    if candidates:
                        expected = candidates[0]
                if os.path.exists(expected):
                    st.image(expected)
                st.subheader("Acurácia por Grupo")
                expected_g = os.path.join(output_dir, f"{prefix}drift_per_group_sync.png")
                if not os.path.exists(expected_g):
                    import glob
                    candidates = glob.glob(os.path.join(output_dir, f"{prefix}drift_per_group_*.png"))
                    if candidates:
                        expected_g = candidates[0]
                if os.path.exists(expected_g):
                    st.image(expected_g)
            else:
                sync_k = sync_key.strip() if sync_key.strip() else None
                async_k = async_key.strip() if async_key.strip() else None
                plot_compare_mode(
                    sync_json.strip(),
                    sync_k,
                    async_json.strip(),
                    async_k,
                    output_dir,
                    alpha=alpha,
                    title=title.strip() or None,
                    include_ema=include_ema,
                    target_accuracies=targets if targets else None,
                    output_prefix=output_prefix.strip(),
                )
                st.success("Graficos gerados.")
                prefix = f"{output_prefix.strip()}_" if output_prefix.strip() else ""
                st.subheader("Comparacao — Acurácia × Tempo")
                compare_png = os.path.join(output_dir, f"{prefix}drift_compare_overall.png")
                if os.path.exists(compare_png):
                    st.image(compare_png)
                st.subheader("Comparacao — Por Grupo")
                group_png = os.path.join(output_dir, f"{prefix}drift_compare_per_group.png")
                if os.path.exists(group_png):
                    st.image(group_png)

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
