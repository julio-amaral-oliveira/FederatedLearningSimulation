from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.e07_drift_agent.plot import (
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
            "initial_rounds": 4,
            "warmup_ticks": 5,
            "monitor_tick_seconds": 10.0,
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
        "clean_evaluations": [
            {"time": onset, "loss": 0.3, "accuracy": 0.6, "stage": "pre_drift"},
            {"time": 140.0, "loss": 0.3, "accuracy": 0.6, "stage": "final"},
        ],
        "clean_training_history": [
            {"time": 25.0, "loss": 0.9, "accuracy": 0.2, "stage": "train_round"},
            {"time": 50.0, "loss": 0.7, "accuracy": 0.35, "stage": "train_round"},
            {"time": 75.0, "loss": 0.5, "accuracy": 0.5, "stage": "train_round"},
            {"time": 100.0, "loss": 0.35, "accuracy": 0.6, "stage": "train_round"},
        ],
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
            trajectory_axis = figure.axes[0]
            self.assertEqual(trajectory_axis.get_xlim(), (0.0, 140.0))
            self.assertEqual(
                trajectory_axis.get_ylabel(), "Accuracy"
            )
            self.assertEqual(
                trajectory_axis.get_xlabel(), "Simulated time (s)"
            )
            trajectory_legend = trajectory_axis.get_legend()
            self.assertIsNotNone(trajectory_legend)
            legend_labels = [text.get_text() for text in trajectory_legend.get_texts()]
            self.assertIn(
                "Baseline mean (dashed, no retraining)",
                legend_labels,
            )
            self.assertIn("Clean accuracy (training)", legend_labels)
            axis_labels = [text.get_text() for text in trajectory_axis.texts]
            self.assertIn("training", axis_labels)
            self.assertIn("warm-up (5 ticks)", axis_labels)
            self.assertIn("drift onset", axis_labels)
            self.assertIn("horizon", axis_labels)
            seed_legend = trajectory_axis.artists[0]
            self.assertIn(
                "Agent (solid, with retraining)",
                [text.get_text() for text in seed_legend.get_texts()],
            )
            self.assertIn(
                "Baseline (dashed, no retraining)",
                [text.get_text() for text in seed_legend.get_texts()],
            )

    def test_trajectory_shows_the_training_phase_band_and_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            figure = build_figure(load_scenario(root))
            trajectory_axis = figure.axes[0]
            spans = [
                patch
                for patch in trajectory_axis.patches
                if patch.get_width() > 0 and patch.get_alpha() != 0.08
            ]
            training_span = next(
                (span for span in spans if span.get_x() == 0.0), None
            )
            self.assertIsNotNone(training_span)
            self.assertEqual(training_span.get_width(), 100.0)
            training_steps = [
                line
                for line in trajectory_axis.lines
                if line.get_linestyle() == "-" and len(line.get_xdata()) == 4
            ]
            self.assertEqual(len(training_steps), 1)
            np.testing.assert_array_equal(
                training_steps[0].get_xdata(),
                [25.0, 50.0, 75.0, 100.0],
            )

    def test_trajectory_without_training_history_marks_clean_accuracy_at_onset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            for arm in ("agent", "baseline"):
                payload_path = root / "seed_42" / f"{arm}.json"
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                del payload["clean_training_history"]
                payload_path.write_text(json.dumps(payload), encoding="utf-8")
            figure = build_figure(load_scenario(root))
            trajectory_legend = figure.axes[0].get_legend()
            legend_labels = [
                text.get_text() for text in trajectory_legend.get_texts()
            ]
            self.assertIn("Clean accuracy at onset", legend_labels)
            self.assertNotIn("Clean accuracy (training)", legend_labels)

    def test_plot_marks_a_pair_whose_retraining_was_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_pair(root, "seed_42", 42)
            agent_path = root / "seed_42" / "agent.json"
            payload = json.loads(agent_path.read_text(encoding="utf-8"))
            payload["retrain_round_events"] = []
            payload["metrics"]["retrain_skipped_budget"] = True
            agent_path.write_text(json.dumps(payload), encoding="utf-8")

            figure = build_figure(load_scenario(root))
            seed_legend = figure.axes[0].artists[0]
            self.assertIn(
                "Seed 42 (no retrain)",
                [text.get_text() for text in seed_legend.get_texts()],
            )
            trajectory_legend = figure.axes[0].get_legend()
            self.assertIn(
                "Decision without retraining (budget)",
                [text.get_text() for text in trajectory_legend.get_texts()],
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
