"""Calibrate corruption severities from one clean federated checkpoint."""

import argparse
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from experiments.result_io import atomic_write_json


CORRUPTIONS = ["gaussian_noise", "frosted_glass_blur", "motion_blur", "fog"]
SEVERITIES = [1, 2, 3, 4, 5]
TARGET_MIN_ACC = 0.25
TARGET_MAX_ACC = 0.50
OUTPUT_BASE_DIR = "output/severity_calibration"


def calibrate_severities(
    corruption_names: Iterable[str],
    severities: Iterable[int],
    evaluate_accuracy: Callable[[str, int], float],
    *,
    minimum_accuracy: float,
    maximum_accuracy: float,
) -> dict:
    """Evaluate every corruption/severity pair and select the lowest valid one.

    The full table is evaluated before selection so the persisted calibration
    remains auditable even when a low severity already meets the target.
    """
    if minimum_accuracy >= maximum_accuracy:
        raise ValueError("minimum_accuracy must be less than maximum_accuracy")

    corruption_names = tuple(corruption_names)
    severities = tuple(severities)
    rows = []
    for corruption in corruption_names:
        for severity in severities:
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": float(evaluate_accuracy(corruption, severity)),
                }
            )

    selected = {}
    for corruption in dict.fromkeys(row["corruption"] for row in rows):
        candidates = [
            row
            for row in rows
            if row["corruption"] == corruption
            and minimum_accuracy < row["accuracy"] < maximum_accuracy
        ]
        choice = min(candidates, key=lambda row: row["severity"], default=None)
        selected[corruption] = (
            {"severity": choice["severity"], "accuracy": choice["accuracy"]}
            if choice is not None
            else None
        )

    return {
        "rows": rows,
        "selected": selected,
        "accuracy_interval": {
            "minimum": minimum_accuracy,
            "maximum": maximum_accuracy,
            "strict": True,
        },
    }


def calibrate_clean_checkpoint(
    config,
    *,
    corruption_names: Iterable[str],
    severities: Iterable[int],
    minimum_accuracy: float,
    maximum_accuracy: float,
    corruption_fn=None,
) -> dict:
    """Train one clean server, then evaluate all corrupted test-set copies."""
    from experiments.smoke_drift import _build_server, _global_weights_digest

    if corruption_fn is None:
        from src.utils.corruptions import apply_corruption

        corruption_fn = apply_corruption

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    server = _build_server(config)
    server.run_rounds(config.initial_rounds, record_default_metrics=False)
    clean_checkpoint_digest = _global_weights_digest(server)
    clean_inputs, clean_labels = server.testing_data

    def evaluate_accuracy(corruption: str, severity: int) -> float:
        # Every evaluation starts from a new copy; the clean checkpoint itself
        # is never altered by a corruption or a monitoring/retraining episode.
        corrupted = corruption_fn(
            torch.as_tensor(clean_inputs).detach().clone(),
            corruption,
            severity,
            seed=config.seed,
        )
        if not isinstance(corrupted, torch.Tensor):
            raise TypeError("corruption_fn must return torch.Tensor")
        if corrupted.shape != clean_inputs.shape:
            raise ValueError("corruption_fn must preserve the testing-data shape")
        _, accuracy = server.evaluate_dataset(
            (corrupted.detach().cpu().numpy().astype(clean_inputs.dtype, copy=False), clean_labels.copy())
        )
        return float(accuracy)

    result = calibrate_severities(
        corruption_names,
        severities,
        evaluate_accuracy,
        minimum_accuracy=minimum_accuracy,
        maximum_accuracy=maximum_accuracy,
    )
    result.update(
        {
            "schema_version": 1,
            "dataset": config.dataset,
            "seed": config.seed,
            "clean_checkpoint_digest": clean_checkpoint_digest,
        }
    )
    return result


def save_calibration(result: dict, output_dir: str) -> str:
    """Persist the complete calibration table and its shared checkpoint ID."""
    path = Path(output_dir) / "severity_calibration.json"
    atomic_write_json(path, result)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate all corruptions from one clean checkpoint")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    parser.add_argument("--severities", nargs="+", type=int, default=SEVERITIES)
    parser.add_argument("--minimum-accuracy", type=float, default=TARGET_MIN_ACC)
    parser.add_argument("--maximum-accuracy", type=float, default=TARGET_MAX_ACC)
    parser.add_argument("--initial-rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=OUTPUT_BASE_DIR)
    args = parser.parse_args()

    from experiments.smoke_drift import DriftEpisodeConfig

    config = DriftEpisodeConfig(
        dataset=args.dataset,
        initial_rounds=args.initial_rounds,
        num_clients=args.num_clients,
        seed=args.seed,
    )
    result = calibrate_clean_checkpoint(
        config,
        corruption_names=args.corruptions,
        severities=args.severities,
        minimum_accuracy=args.minimum_accuracy,
        maximum_accuracy=args.maximum_accuracy,
    )
    print(save_calibration(result, args.output_dir))


if __name__ == "__main__":
    main()
