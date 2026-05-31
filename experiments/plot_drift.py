import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


def _parse_key_meta(key):
    m = re.match(r"T_drift_(\d+)_p(\d+)_(\w+)", key)
    if m:
        return int(m.group(1)), m.group(3)
    m = re.match(r"T_drift_(\d+)_(\w+)", key)
    if m:
        return int(m.group(1)), m.group(2)
    return None, "unknown"


def _extract_group_names(entries):
    for e in entries:
        for k in e:
            if k.startswith("accuracy_") and k not in ("accuracy",):
                return [k.replace("accuracy_", "") for k in e if k.startswith("accuracy_")]
    return []


def _load_json(filepath, key=None):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {filepath}")
        return data[key]
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]]
    raise KeyError(
        f"Multiple keys in {filepath}: {keys}. Use --key to specify."
    )


def _time_to_acc(points_sorted, target):
    for e in points_sorted:
        if e["accuracy"] >= target:
            return e["time"]
    return None


def _draw_target_lines(ax, target_accuracies):
    for target in target_accuracies:
        ax.axhline(y=target, color="black", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(ax.get_xlim()[1], target, f" acc={target:.0%}",
                ha="right", va="bottom", fontsize=8, color="black", alpha=0.6)


def _compute_metrics(entries):
    from collections import OrderedDict

    points = sorted(entries, key=lambda e: e["time"])
    times = np.array([e["time"] for e in points])
    accs = np.array([e["accuracy"] for e in points])
    phases = [e["phase"] for e in points]

    result = OrderedDict()
    result["n_evals"] = len(points)
    result["time_total_s"] = times[-1] if len(times) > 0 else 0
    result["acc_initial"] = accs[0] if len(accs) > 0 else 0
    result["acc_final"] = accs[-1] if len(accs) > 0 else 0
    result["max_acc"] = np.max(accs) if len(accs) > 0 else 0
    result["avg_last10"] = np.mean(accs[-10:]) if len(accs) >= 10 else np.mean(accs)
    result["tail_std"] = np.std(accs[-10:]) if len(accs) >= 10 else 0
    result["n_phases"] = len(set(phases))
    result["sustained_acc"] = np.mean(accs) if len(accs) > 0 else 0

    return result, points


def _print_metrics(label, metrics):
    print(f"\n--- Metricas: {label} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25} {value:>10.4f}")
        else:
            print(f"  {key:<25} {value:>10}")


def _print_time_to_acc(label, points_sorted, target_accuracies):
    if not target_accuracies:
        return
    print(f"\n  Tempo ate acuracia ({label}):")
    for target in target_accuracies:
        t = _time_to_acc(points_sorted, target)
        if t is not None:
            print(f"    acc={target:.0%}  ->  {t:.1f}s")
        else:
            print(f"    acc={target:.0%}  ->  nao atingido")


def plot_single_mode(filepath, key, output_dir, alpha=0.1, title=None,
                     include_ema=True, target_accuracies=None,
                     output_prefix=""):
    """Plot accuracy × time for a single mode with drift phase lines."""
    entries = _load_json(filepath, key)
    T_drift, mode_label = _parse_key_meta(key or list(json.load(open(filepath)))[0])
    group_names = _extract_group_names(entries)

    points = sorted(entries, key=lambda e: e["time"])
    times = np.array([e["time"] for e in points]) / 60.0
    acc = np.array([e["accuracy"] for e in points])
    smoothed = exponential_moving_average(acc, alpha)

    phases = sorted(set(e["phase"] for e in points))
    max_phase = max(phases)
    drift_times_min = [(i * T_drift) / 60.0 for i in range(1, max_phase)]

    prefix = f"{output_prefix}_" if output_prefix else ""

    plt = _get_plt()
    raw_lw = 2 if not include_ema else 1

    # ---- Plot 1: Overall accuracy ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, acc, alpha=0.5, linewidth=raw_lw, color="#1f77b4",
            label=f"{mode_label} (bruto)")
    if include_ema:
        ax.plot(times, smoothed, linewidth=2, color="#1f77b4",
                label=f"{mode_label} (EMA, a={alpha})")

    for dt in drift_times_min:
        ax.axvline(x=dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(dt, ax.get_ylim()[0], f"drift\nt={dt*60:.0f}s",
                ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Tempo virtual (min)", fontsize=12)
    ax.set_ylabel("Acurácia (teste completo)", fontsize=12)
    ax.set_title(title or f"Temporal Drift — {mode_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{prefix}drift_overall_{mode_label}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Grafico salvo: {png_path}")
    plt.close(fig)

    # ---- Plot 2: Per-group accuracy ----
    if group_names:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#ff7f0e", "#2ca02c"]
        for gi, gname in enumerate(group_names):
            gkey = f"accuracy_{gname}"
            gacc = np.array([e[gkey] for e in points])
            gsmoothed = exponential_moving_average(gacc, alpha)
            ax.plot(times, gacc, alpha=0.4, linewidth=raw_lw, color=colors[gi],
                    label=f"Grupo {gname} (bruto)")
            if include_ema:
                ax.plot(times, gsmoothed, linewidth=2, color=colors[gi],
                        label=f"Grupo {gname} (EMA)")
        for dt in drift_times_min:
            ax.axvline(x=dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Tempo virtual (min)", fontsize=12)
        ax.set_ylabel("Acurácia por grupo", fontsize=12)
        ax.set_title(title or f"Temporal Drift — {mode_label} — Por Grupo", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(output_dir, f"{prefix}drift_per_group_{mode_label}.png")
        fig.savefig(png_path, dpi=150)
        print(f"Grafico salvo: {png_path}")
        plt.close(fig)

    # ---- Metrics ----
    metrics, pts = _compute_metrics(entries)
    _print_metrics(mode_label, metrics)
    _print_time_to_acc(mode_label, pts, target_accuracies)


def plot_compare_mode(sync_filepath, sync_key, async_filepath, async_key,
                      output_dir, alpha=0.1, title=None, include_ema=True,
                      target_accuracies=None, output_prefix=""):
    """Compare sync vs async drift curves."""
    sync_entries = _load_json(sync_filepath, sync_key)
    async_entries = _load_json(async_filepath, async_key)

    sync_key_raw = sync_key or list(json.load(open(sync_filepath)))[0]
    async_key_raw = async_key or list(json.load(open(async_filepath)))[0]
    T_drift_s, _ = _parse_key_meta(sync_key_raw)
    T_drift_a, _ = _parse_key_meta(async_key_raw)
    T_drift = T_drift_s or T_drift_a

    group_names_s = _extract_group_names(sync_entries)
    group_names_a = _extract_group_names(async_entries)
    group_names = group_names_s or group_names_a

    prefix = f"{output_prefix}_" if output_prefix else ""

    plt = _get_plt()
    raw_lw = 2 if not include_ema else 1

    # ---- Plot 1: Overall accuracy comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))

    for entries, label, color in [
        (sync_entries, "Sync", "#1f77b4"),
        (async_entries, "Async", "#ff7f0e"),
    ]:
        points = sorted(entries, key=lambda e: e["time"])
        times = np.array([e["time"] for e in points]) / 60.0
        acc = np.array([e["accuracy"] for e in points])
        smoothed = exponential_moving_average(acc, alpha)
        ax.plot(times, acc, alpha=0.4, linewidth=raw_lw, color=color,
                label=f"{label} (bruto)")
        if include_ema:
            ax.plot(times, smoothed, linewidth=2, color=color,
                    label=f"{label} (EMA, a={alpha})")

    if T_drift:
        phases = sorted(set(
            e["phase"] for e in (sync_entries + async_entries)
        ))
        max_phase = max(phases)
        for i in range(1, max_phase):
            dt = (i * T_drift) / 60.0
            ax.axvline(x=dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    if target_accuracies:
        _draw_target_lines(ax, target_accuracies)

    ax.set_xlabel("Tempo virtual (min)", fontsize=12)
    ax.set_ylabel("Acurácia (teste completo)", fontsize=12)
    ax.set_title(title or "Sync vs Async — Temporal Drift", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{prefix}drift_compare_overall.png")
    fig.savefig(png_path, dpi=150)
    print(f"Grafico salvo: {png_path}")
    plt.close(fig)

    # ---- Plot 2: Per-group comparison ----
    if group_names:
        fig, axes = plt.subplots(len(group_names), 1, figsize=(12, 5 * len(group_names)))
        if len(group_names) == 1:
            axes = [axes]

        for gi, gname in enumerate(group_names):
            ax = axes[gi]
            gkey = f"accuracy_{gname}"
            for entries, label, color in [
                (sync_entries, "Sync", "#1f77b4"),
                (async_entries, "Async", "#ff7f0e"),
            ]:
                points = sorted(entries, key=lambda e: e["time"])
                times = np.array([e["time"] for e in points]) / 60.0
                gacc = np.array([e[gkey] for e in points])
                gsmoothed = exponential_moving_average(gacc, alpha)
                ax.plot(times, gacc, alpha=0.4, linewidth=raw_lw, color=color,
                        label=f"{label} (bruto)")
                if include_ema:
                    ax.plot(times, gsmoothed, linewidth=2, color=color,
                            label=f"{label} (EMA)")

            if T_drift:
                for i in range(1, max_phase):
                    dt = (i * T_drift) / 60.0
                    ax.axvline(x=dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Tempo virtual (min)", fontsize=12)
            ax.set_ylabel(f"Acurácia Grupo {gname}", fontsize=12)
            ax.set_title(f"Grupo {gname}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title or "Sync vs Async — Per-Group Accuracy", fontsize=13)
        fig.tight_layout()
        png_path = os.path.join(output_dir, f"{prefix}drift_compare_per_group.png")
        fig.savefig(png_path, dpi=150)
        print(f"Grafico salvo: {png_path}")
        plt.close(fig)

    # ---- Metrics ----
    sync_metrics, sync_pts = _compute_metrics(sync_entries)
    async_metrics, async_pts = _compute_metrics(async_entries)

    print("\n===== METRICAS COMPARATIVAS =====")
    header = f"{'Metrica':<25} {'Sync':>12} {'Async':>12}"
    print(header)
    print("-" * len(header))
    for key in sync_metrics:
        sv = sync_metrics[key]
        av = async_metrics[key]
        if isinstance(sv, float):
            print(f"{key:<25} {sv:>12.4f} {av:>12.4f}")
        else:
            print(f"{key:<25} {sv:>12} {av:>12}")

    if target_accuracies:
        print(f"\n  Tempo ate acuracia:")
        header2 = f"  {'Alvo':<10} {'Sync':>12} {'Async':>12}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for target in target_accuracies:
            st = _time_to_acc(sync_pts, target)
            at = _time_to_acc(async_pts, target)
            st_str = f"{st:.1f}s" if st is not None else "nao atingido"
            at_str = f"{at:.1f}s" if at is not None else "nao atingido"
            print(f"  acc={target:.0%}  {st_str:>12} {at_str:>12}")


def _parse_float_list(text):
    if not text or not text.strip():
        return []
    values = []
    for raw in text.replace("\n", ",").split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Plot temporal drift results"
    )
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "compare"])
    parser.add_argument("--json", type=str, help="JSON file for single mode")
    parser.add_argument("--key", type=str, default=None,
                        help="Key inside JSON (optional if only one key)")
    parser.add_argument("--sync-json", type=str,
                        help="Sync JSON file for compare mode")
    parser.add_argument("--async-json", type=str,
                        help="Async JSON file for compare mode")
    parser.add_argument("--sync-key", type=str, default=None)
    parser.add_argument("--async-key", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for plots")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="EMA smoothing factor")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--target-accuracy", type=str, default=None,
                        help="Target accuracies, comma-separated (ex: 0.50,0.60)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA curve")
    parser.add_argument("--output-prefix", type=str, default="",
                        help="Prefix for output filenames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    targets = _parse_float_list(args.target_accuracy)
    include_ema = not args.no_ema

    if args.mode == "single":
        if not args.json:
            parser.error("--json required for single mode")
        plot_single_mode(
            args.json, args.key, args.output_dir,
            alpha=args.alpha, title=args.title,
            include_ema=include_ema,
            target_accuracies=targets if targets else None,
            output_prefix=args.output_prefix,
        )
    elif args.mode == "compare":
        if not args.sync_json or not args.async_json:
            parser.error("--sync-json and --async-json required for compare mode")
        plot_compare_mode(
            args.sync_json, args.sync_key,
            args.async_json, args.async_key,
            args.output_dir, alpha=args.alpha, title=args.title,
            include_ema=include_ema,
            target_accuracies=targets if targets else None,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()

