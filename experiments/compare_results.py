"""Compare federated learning result curves from explicit JSON scenarios.

Example:
    python experiments/compare_results.py ^
      --scenario "Sync IID p75=output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_sync.json#75" ^
      --scenario "Async IID p75=output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_async.json#75" ^
      --target-accuracy 0.50 ^
      --target-accuracy 0.60 ^
      --horizon-seconds 4000 ^
      --horizon-seconds 5000 ^
      --title "Sync vs Async - CIFAR-10 IID p75" ^
      --output experiments/comparison_iid_p75_5000
"""

import argparse
import sys
from pathlib import Path

try:
    from experiments.comparison_core import compare_scenarios, parse_scenario_spec
except ModuleNotFoundError:
    from comparison_core import compare_scenarios, parse_scenario_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more FL result JSON scenarios and always generate "
            "Markdown, CSV, and PNG outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        metavar="LABEL=PATH[#KEY]",
        help=(
            "Explicit scenario to compare. Use '#KEY' for JSON files with "
            "multiple percentile keys, e.g. "
            "'Async IID=output-cifar-10/result.json#75'. Repeat at least twice."
        ),
    )
    parser.add_argument(
        "--target-accuracy",
        action="append",
        type=float,
        default=[],
        help=(
            "Accuracy target applied to every scenario. Repeat for multiple "
            "targets, e.g. --target-accuracy 0.50 --target-accuracy 0.60."
        ),
    )
    parser.add_argument(
        "--horizon-seconds",
        action="append",
        type=float,
        default=[],
        help=(
            "Simulated-time horizon applied to every scenario. Repeat for "
            "multiple horizons, e.g. --horizon-seconds 4000."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output base path without extension. The command writes .md, .csv, "
            "and .png files."
        ),
    )
    parser.add_argument(
        "--title",
        default="Comparacao de Resultados",
        help="Title used in the Markdown report and PNG chart.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA alpha used for the smoothed curve in the PNG.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the smoothed EMA curve and plot only raw accuracy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        scenario_specs = [parse_scenario_spec(value) for value in args.scenario]
        result = compare_scenarios(
            scenario_specs,
            args.output,
            target_accuracies=args.target_accuracy,
            horizon_seconds=args.horizon_seconds,
            ema_alpha=args.ema_alpha,
            include_ema=not args.no_ema,
            title=args.title,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Markdown: {result['markdown']}")
    print(f"CSV:      {result['csv']}")
    print(f"PNG:      {result['png']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
