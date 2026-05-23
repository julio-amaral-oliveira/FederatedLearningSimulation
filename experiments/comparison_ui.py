"""Small Streamlit UI for the FL result comparator.

Run:
    streamlit run experiments/comparison_ui.py
"""

from pathlib import Path

try:
    from experiments.comparison_core import (
        ScenarioSpec,
        compare_scenarios,
        parse_scenario_spec,
    )
except ModuleNotFoundError:
    from comparison_core import ScenarioSpec, compare_scenarios, parse_scenario_spec


DEFAULT_SCENARIO_ROWS = [
    {
        "label": "Sync IID p75",
        "path": "output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_sync.json",
        "key": "75",
    },
    {
        "label": "Async IID p75",
        "path": "output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_async.json",
        "key": "75",
    },
]


def parse_float_list(text: str) -> list[float]:
    values = []
    normalized = text.replace("\n", ",")
    for raw_value in normalized.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        values.append(float(raw_value))
    return values


def parse_scenario_lines(text: str):
    specs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        specs.append(parse_scenario_spec(line))
    return specs


def build_scenario_specs(rows: list[dict[str, str]]) -> list[ScenarioSpec]:
    specs = []
    for index, row in enumerate(rows, start=1):
        label = row.get("label", "").strip()
        path = row.get("path", "").strip()
        key = row.get("key", "").strip() or None

        if not label and not path and key is None:
            continue
        if not label:
            raise ValueError(f"Cenario {index}: label nao pode ficar vazio")
        if not path:
            raise ValueError(f"Cenario {index}: arquivo nao pode ficar vazio")

        specs.append(ScenarioSpec(label=label, path=Path(path), key=key))
    return specs


def rows_to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "Sem metricas para exibir."

    fieldnames = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [_markdown_value(row.get(field)) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Comparacao FL", layout="wide")
    st.title("Comparacao de Resultados FL")

    st.write(
        "Informe dois ou mais cenarios. A UI gera os mesmos arquivos do CLI: "
        "Markdown, CSV e PNG."
    )

    title = st.text_input(
        "Titulo",
        value="Comparacao de Resultados",
        help="Aparece no topo do Markdown e como titulo do grafico.",
    )

    scenario_count = st.number_input(
        "Numero de cenarios",
        min_value=2,
        max_value=12,
        value=2,
        step=1,
    )

    scenario_rows = []
    st.subheader("Cenarios")
    for index in range(int(scenario_count)):
        defaults = (
            DEFAULT_SCENARIO_ROWS[index]
            if index < len(DEFAULT_SCENARIO_ROWS)
            else {"label": "", "path": "", "key": ""}
        )
        columns = st.columns([2, 5, 1])
        with columns[0]:
            label = st.text_input(
                "Label",
                value=defaults["label"],
                key=f"scenario_label_{index}",
            )
        with columns[1]:
            path = st.text_input(
                "Arquivo JSON",
                value=defaults["path"],
                key=f"scenario_path_{index}",
            )
        with columns[2]:
            key = st.text_input(
                "Chave",
                value=defaults["key"],
                key=f"scenario_key_{index}",
                help="Opcional quando o JSON tem uma unica chave ou e uma lista direta.",
            )
        scenario_rows.append({"label": label, "path": path, "key": key})

    col_left, col_right = st.columns(2)
    with col_left:
        target_text = st.text_input(
            "Alvos de acuracia",
            value="0.50, 0.60",
            help="Valores separados por virgula ou quebra de linha. Pode deixar vazio.",
        )
        ema_alpha = st.number_input(
            "EMA alpha",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
        )
    with col_right:
        horizon_text = st.text_input(
            "Horizontes em segundos",
            value="4000, 5000",
            help="Valores separados por virgula ou quebra de linha. Pode deixar vazio.",
        )
        include_ema = st.checkbox("Mostrar curva EMA", value=True)

    output_base = st.text_input(
        "Saida",
        value="output/comparison_ui_output",
        help="Caminho base sem extensao. Serao gerados .md, .csv e .png.",
    )

    if st.button("Gerar comparacao", type="primary"):
        try:
            scenario_specs = build_scenario_specs(scenario_rows)
            target_accuracies = parse_float_list(target_text)
            horizon_seconds = parse_float_list(horizon_text)

            result = compare_scenarios(
                scenario_specs,
                Path(output_base),
                target_accuracies=target_accuracies,
                horizon_seconds=horizon_seconds,
                ema_alpha=ema_alpha,
                include_ema=include_ema,
                title=title.strip() or "Comparacao de Resultados",
            )
        except Exception as exc:
            st.error(str(exc))
            return

        st.success("Comparacao gerada.")
        st.code(
            "\n".join(
                [
                    f"Markdown: {result['markdown']}",
                    f"CSV:      {result['csv']}",
                    f"PNG:      {result['png']}",
                ]
            )
        )

        st.subheader("Grafico")
        st.image(str(result["png"]))

        st.subheader("Metricas")
        st.markdown(rows_to_markdown_table(result["rows"]))

        st.subheader("Markdown")
        st.markdown(Path(result["markdown"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
