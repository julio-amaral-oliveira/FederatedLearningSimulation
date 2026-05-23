import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

REQUIRED_ENTRY_FIELDS = ("accuracy", "loss", "time")


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    path: Path
    key: str | None = None


@dataclass(frozen=True)
class Scenario:
    label: str
    path: Path
    key: str | None
    entries: list[dict]


def parse_scenario_spec(value: str) -> ScenarioSpec:
    if "=" not in value:
        raise ValueError(
            "scenario must use 'Label=path.json#key', for example "
            "'Async IID=output-cifar-10/result.json#75'"
        )

    label, location = value.split("=", 1)
    label = label.strip()
    location = location.strip()
    if not label:
        raise ValueError("scenario label cannot be empty")
    if not location:
        raise ValueError(f"scenario '{label}' must include a JSON path")

    path_text, key = _split_path_key(location)
    return ScenarioSpec(label=label, path=Path(path_text), key=key)


def load_scenario(spec: ScenarioSpec) -> Scenario:
    if not spec.path.exists():
        raise ValueError(f"scenario '{spec.label}' file does not exist: {spec.path}")

    with spec.path.open(encoding="utf-8") as file:
        payload = json.load(file)

    entries, key = _select_entries(payload, spec)
    validated_entries = _validate_entries(entries, spec.label)
    return Scenario(spec.label, spec.path, key, validated_entries)


def compute_metrics(
    entries: list[dict],
    target_accuracies: list[float] | None = None,
    horizon_seconds: list[float] | None = None,
) -> dict:
    if not entries:
        raise ValueError("metrics require at least one entry")

    sorted_entries = _sort_entries(entries)
    accuracies = [float(entry["accuracy"]) for entry in sorted_entries]
    losses = [float(entry["loss"]) for entry in sorted_entries]
    times = [float(entry["time"]) for entry in sorted_entries]
    tail_n = max(1, len(accuracies) // 5)
    max_acc = max(accuracies)
    final_acc = accuracies[-1]

    metrics = {
        "n_evals": len(sorted_entries),
        "time_total_min": times[-1] / 60.0,
        "acc_initial": accuracies[0],
        "acc_final": final_acc,
        "max_acc": max_acc,
        "avg_last10": sum(accuracies[-min(10, len(accuracies)) :])
        / min(10, len(accuracies)),
        "tail_std": _std(accuracies[-tail_n:]),
        "delta_peak_final": max_acc - final_acc,
        "loss_final": losses[-1],
        "time_to_90pct_peak": time_to_target_accuracy(
            sorted_entries, 0.9 * max_acc
        ),
        "area_under_accuracy_time": area_under_accuracy_time(sorted_entries),
    }

    for target in target_accuracies or []:
        metrics[f"time_to_acc_{format_metric_suffix(target * 100)}"] = (
            time_to_target_accuracy(sorted_entries, target)
        )

    for horizon in horizon_seconds or []:
        suffix = format_metric_suffix(horizon)
        metrics[f"acc_at_{suffix}s"] = accuracy_at_horizon(sorted_entries, horizon)
        metrics[f"max_acc_until_{suffix}s"] = max_accuracy_until(
            sorted_entries, horizon
        )

    return metrics


def compare_scenarios(
    scenario_specs: list[ScenarioSpec],
    output_base: Path,
    target_accuracies: list[float] | None = None,
    horizon_seconds: list[float] | None = None,
    ema_alpha: float = 0.1,
    include_ema: bool = True,
    title: str = "Comparacao de Resultados",
) -> dict:
    if len(scenario_specs) < 2:
        raise ValueError("comparison requires at least two scenarios")

    scenarios = [load_scenario(spec) for spec in scenario_specs]
    rows = [
        {
            "scenario": scenario.label,
            "path": str(scenario.path),
            "key": scenario.key or "",
            **compute_metrics(
                scenario.entries,
                target_accuracies=target_accuracies,
                horizon_seconds=horizon_seconds,
            ),
        }
        for scenario in scenarios
    ]

    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")
    png_path = output_base.with_suffix(".png")

    write_csv(rows, csv_path)
    write_markdown(
        scenarios,
        rows,
        md_path,
        target_accuracies=target_accuracies or [],
        horizon_seconds=horizon_seconds or [],
        title=title,
    )
    plot_accuracy(
        scenarios,
        png_path,
        target_accuracies=target_accuracies or [],
        horizon_seconds=horizon_seconds or [],
        ema_alpha=ema_alpha,
        include_ema=include_ema,
        title=title,
    )

    return {"csv": csv_path, "markdown": md_path, "png": png_path, "rows": rows}


def time_to_target_accuracy(entries: list[dict], target_accuracy: float) -> float | None:
    for entry in _sort_entries(entries):
        if float(entry["accuracy"]) >= target_accuracy:
            return float(entry["time"])
    return None


def accuracy_at_horizon(entries: list[dict], horizon_seconds: float) -> float | None:
    eligible = [
        entry for entry in _sort_entries(entries) if float(entry["time"]) <= horizon_seconds
    ]
    if not eligible:
        return None
    return float(eligible[-1]["accuracy"])


def max_accuracy_until(entries: list[dict], horizon_seconds: float) -> float | None:
    eligible = [
        float(entry["accuracy"])
        for entry in _sort_entries(entries)
        if float(entry["time"]) <= horizon_seconds
    ]
    if not eligible:
        return None
    return max(eligible)


def area_under_accuracy_time(entries: list[dict]) -> float:
    sorted_entries = _sort_entries(entries)
    if len(sorted_entries) < 2:
        return 0.0

    area = 0.0
    for previous, current in zip(sorted_entries, sorted_entries[1:]):
        previous_time = float(previous["time"])
        current_time = float(current["time"])
        width = current_time - previous_time
        if width < 0:
            raise ValueError("entries must be sorted by non-decreasing time")
        height = (float(previous["accuracy"]) + float(current["accuracy"])) / 2.0
        area += width * height
    return area


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = _ordered_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def write_markdown(
    scenarios: list[Scenario],
    rows: list[dict],
    output_path: Path,
    target_accuracies: list[float],
    horizon_seconds: list[float],
    title: str = "Comparacao de Resultados",
) -> None:
    lines = [
        f"# {title}",
        "",
        "## Entradas",
        "",
    ]
    for scenario in scenarios:
        key_text = f"#{scenario.key}" if scenario.key is not None else ""
        lines.append(f"- {scenario.label}: `{scenario.path}{key_text}`")

    lines.extend(
        [
            "",
            "## Parametros",
            "",
            f"- Alvos de acuracia: {_list_or_na([_format_percent(v) for v in target_accuracies])}",
            f"- Horizontes: {_list_or_na([f'{format_metric_suffix(v)}s' for v in horizon_seconds])}",
            "",
            "## Metricas",
            "",
        ]
    )
    lines.extend(_markdown_table(rows))
    lines.extend(["", "## Comparacoes Objetivas", ""])
    lines.extend(_objective_comparisons(rows, target_accuracies, horizon_seconds))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy(
    scenarios: list[Scenario],
    output_path: Path,
    target_accuracies: list[float],
    horizon_seconds: list[float],
    ema_alpha: float = 0.1,
    include_ema: bool = True,
    title: str = "Comparacao de Resultados",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, scenario in enumerate(scenarios):
        entries = _sort_entries(scenario.entries)
        times = [float(entry["time"]) for entry in entries]
        accuracies = [float(entry["accuracy"]) for entry in entries]
        color = color_cycle[index % len(color_cycle)]

        axis.plot(
            times,
            accuracies,
            color=color,
            alpha=0.25 if include_ema else 0.9,
            linewidth=0.8,
            label=f"{scenario.label} bruto" if include_ema else scenario.label,
        )
        if include_ema:
            axis.plot(
                times,
                exponential_moving_average(accuracies, ema_alpha),
                color=color,
                linewidth=2.0,
                label=f"{scenario.label} EMA",
            )

    for target in target_accuracies:
        axis.axhline(
            target,
            color="#666666",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label=f"Alvo {_format_percent(target)}",
        )
    for horizon in horizon_seconds:
        axis.axvline(
            horizon,
            color="#888888",
            linestyle=":",
            linewidth=1.1,
            alpha=0.75,
            label=f"Horizonte {format_metric_suffix(horizon)}s",
        )
        axis.text(
            horizon,
            0.02,
            f"{format_metric_suffix(horizon)}s",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#666666",
        )

    axis.set_title(title)
    axis.set_xlabel("Tempo simulado (s)")
    axis.set_ylabel("Acuracia")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def exponential_moving_average(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    if alpha <= 0 or alpha > 1:
        raise ValueError("ema_alpha must be in the interval (0, 1]")

    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1 - alpha) * smoothed[-1])
    return smoothed


def format_metric_suffix(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "_")


def _split_path_key(location: str) -> tuple[str, str | None]:
    if "#" not in location:
        return location, None
    path_text, key = location.rsplit("#", 1)
    path_text = path_text.strip()
    key = key.strip()
    if not path_text:
        raise ValueError("scenario path cannot be empty")
    if not key:
        raise ValueError("scenario key after '#' cannot be empty")
    return path_text, key


def _select_entries(payload, spec: ScenarioSpec) -> tuple[list[dict], str | None]:
    if isinstance(payload, list):
        if spec.key is not None:
            raise ValueError(
                f"scenario '{spec.label}' points to a list JSON; remove '#{spec.key}'"
            )
        return payload, None

    if not isinstance(payload, dict):
        raise ValueError(f"scenario '{spec.label}' JSON must be an object or a list")

    keys = list(payload.keys())
    if spec.key is not None:
        if spec.key not in payload:
            raise ValueError(
                f"scenario '{spec.label}' key '{spec.key}' not found. "
                f"Available keys: {', '.join(keys)}"
            )
        return payload[spec.key], spec.key

    if len(keys) == 1:
        key = keys[0]
        return payload[key], key

    raise ValueError(
        f"scenario '{spec.label}' JSON has multiple keys "
        f"({', '.join(keys)}); use path.json#key"
    )


def _validate_entries(entries, label: str) -> list[dict]:
    if not isinstance(entries, list):
        raise ValueError(f"scenario '{label}' selected data must be a list")
    if not entries:
        raise ValueError(f"scenario '{label}' selected data is empty")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"scenario '{label}' entry {index} must be an object")
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"scenario '{label}' entry {index} is missing: {', '.join(missing)}"
            )
        for field in REQUIRED_ENTRY_FIELDS:
            try:
                float(entry[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"scenario '{label}' entry {index} field '{field}' must be numeric"
                ) from exc
    return entries


def _sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda entry: float(entry["time"]))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _ordered_fieldnames(rows: list[dict]) -> list[str]:
    preferred = [
        "scenario",
        "path",
        "key",
        "n_evals",
        "time_total_min",
        "acc_initial",
        "acc_final",
        "max_acc",
        "avg_last10",
        "tail_std",
        "delta_peak_final",
        "loss_final",
        "time_to_90pct_peak",
        "area_under_accuracy_time",
    ]
    seen = set()
    fieldnames = []
    for field in preferred:
        if any(field in row for row in rows):
            fieldnames.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    return fieldnames


def _markdown_table(rows: list[dict]) -> list[str]:
    fieldnames = _ordered_fieldnames(rows)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [_markdown_value(row.get(field)) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _objective_comparisons(
    rows: list[dict], target_accuracies: list[float], horizon_seconds: list[float]
) -> list[str]:
    comparisons = []
    comparisons.append(_best_row_line(rows, "max_acc", "Maior max_acc", higher=True))
    comparisons.append(
        _best_row_line(rows, "acc_final", "Maior acc_final", higher=True)
    )
    comparisons.append(
        _best_row_line(
            rows, "delta_peak_final", "Menor queda pico-final", higher=False
        )
    )

    for target in target_accuracies:
        field = f"time_to_acc_{format_metric_suffix(target * 100)}"
        comparisons.append(
            _best_row_line(
                rows,
                field,
                f"Menor tempo ate {_format_percent(target)}",
                higher=False,
            )
        )

    for horizon in horizon_seconds:
        suffix = format_metric_suffix(horizon)
        comparisons.append(
            _best_row_line(
                rows,
                f"max_acc_until_{suffix}s",
                f"Maior max_acc ate {suffix}s",
                higher=True,
            )
        )
    return comparisons


def _best_row_line(
    rows: list[dict], field: str, label: str, higher: bool
) -> str:
    candidates = [row for row in rows if row.get(field) is not None]
    if not candidates:
        return f"- {label}: N/A"
    best = max(candidates, key=lambda row: row[field]) if higher else min(
        candidates, key=lambda row: row[field]
    )
    return f"- {label}: {best['scenario']} ({_markdown_value(best[field])})"


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _markdown_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _format_percent(value: float) -> str:
    return f"{value * 100:g}%"


def _list_or_na(values: list[str]) -> str:
    return ", ".join(values) if values else "N/A"
