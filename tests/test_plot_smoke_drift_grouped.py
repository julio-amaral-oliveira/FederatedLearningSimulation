from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.plot_smoke_drift import (
    ScenarioError,
    build_figure,
    forward_fill_steps,
    load_scenario,
)


def _payload(seed: int, baseline: bool, *, tau: float = 0.5) -> dict:
    onset = 100.0
    return {
        "experiment_config": {
            "corruption": "identity",
            "severity": 0,
            "seed": seed,
            "tau": tau,
            "baseline": baseline,
            "production_horizon_seconds": 40.0,
        },
        "metadata": {
            "corruption": "identity",
            "severity": 0,
            "seed": seed,
            "tau": tau,
            "baseline": baseline,
            "production_start_time": onset,
            "end_time_seconds": 140.0,
            "production_horizon_seconds": 40.0,
        },
        "metrics": {
            "downtime_seconds": 0.0,
            "clean_retention_delta": 0.0 if baseline else 0.02,
        },
        "corrupted_accuracy_history": [
            {"time": onset, "accuracy": 0.6, "stage": "drift_onset"},
            {"time": 110.0, "accuracy": 0.6, "stage": "monitor_tick"},
            {
                "time": 140.0,
                "accuracy": 0.6 if baseline else 0.7,
                "stage": "episode_end",
            },
        ],
        "drift_events": [{"time": 110.0, "client_id": "oracle", "score": 1.0}],
        "retrain_decisions": [] if baseline else [{"time": 110.0}],
        "retrain_round_events": (
            [] if baseline else [{"started_time": 110.0, "completed_time": 120.0}]
        ),
    }


def _write_pair(root: Path, directory: str, seed: int, *, agent_tau: float = 0.5) -> None:
    target = root / directory
    target.mkdir()
    (target / "agent.json").write_text(
        json.dumps(_payload(seed, False, tau=agent_tau)), encoding="utf-8"
    )
    (target / "baseline.json").write_text(
        json.dumps(_payload(seed, True)), encoding="utf-8"
    )


class PlotSmokeDriftTests(unittest.TestCase):
    def test_forward_fill_aligns_union_without_interpolation(self) -> None:
        union, aligned = forward_fill_steps(
            [np.array([0.0, 10.0]), np.array([0.0, 5.0, 10.0])],
            [np.array([0.2, 0.8]), np.array([0.4, 0.5, 0.6])],
        )
        np.testing.assert_array_equal(union, [0.0, 5.0, 10.0])
        np.testing.assert_array_equal(aligned[0], [0.2, 0.2, 0.8])

    def test_loads_pairs_and_marks_oracle_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            _write_pair(root, "seed_43", 43)
            pairs = load_scenario(root)
            figure = build_figure(pairs)
            self.assertEqual([pair.seed for pair in pairs], [42, 43])
            self.assertIn("oracle-trigger control", figure._suptitle.get_text())
            trajectory_legend = figure.axes[0].get_legend()
            self.assertIsNotNone(trajectory_legend)
            self.assertIn(
                "Baseline mean (dashed, no retraining)",
                [text.get_text() for text in trajectory_legend.get_texts()],
            )
            seed_legend = figure.axes[0].artists[0]
            self.assertIn(
                "Agent (solid, with retraining)",
                [text.get_text() for text in seed_legend.get_texts()],
            )
            self.assertIn(
                "Baseline (dashed, no retraining)",
                [text.get_text() for text in seed_legend.get_texts()],
            )

    def test_zero_downtime_uses_horizon_scale_and_keeps_both_bars_visible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            figure = build_figure(load_scenario(root))
            downtime_axis = figure.axes[2]
            self.assertEqual(downtime_axis.get_xlim(), (0.0, 48.0))

            self.assertEqual(len(downtime_axis.patches), 4)
            self.assertEqual(downtime_axis.patches[0].get_hatch(), "//")
            self.assertIsNone(downtime_axis.patches[1].get_hatch())
            self.assertEqual(downtime_axis.patches[2].get_hatch(), "//")
            self.assertIsNone(downtime_axis.patches[3].get_hatch())

    def test_clean_retention_uses_two_percent_decimals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            figure = build_figure(load_scenario(root))
            retention_axis = figure.axes[3]
            self.assertEqual(retention_axis.xaxis.get_major_formatter().decimals, 2)

    def test_rejects_missing_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "seed_42"
            target.mkdir()
            (target / "agent.json").write_text(
                json.dumps(_payload(42, False)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ScenarioError, "missing arm"):
                load_scenario(root)

    def test_rejects_duplicate_numeric_seed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_1", 1)
            _write_pair(root, "seed_01", 1)
            with self.assertRaisesRegex(ScenarioError, "duplicate seed"):
                load_scenario(root)

    def test_rejects_incompatible_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42, agent_tau=0.4)
            with self.assertRaisesRegex(ScenarioError, "incompatible paired arms"):
                load_scenario(root)


if __name__ == "__main__":
    unittest.main()
