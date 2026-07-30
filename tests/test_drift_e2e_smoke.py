import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch.nn as nn

from experiments.drift_results import validate_pair
from experiments.plot_smoke_drift import plot_scenario
from experiments.result_io import load_persisted_pair
from experiments.run_smoke_drift import run_matrix
from experiments.smoke_drift import DriftEpisodeConfig, run_drift_comparison


class _SmokeClient:
    def __init__(self) -> None:
        self.dataset = (
            np.zeros((2, 1, 2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        )


class _SmokeServer:
    def __init__(self) -> None:
        self.clients = [_SmokeClient()]
        self.testing_data = (
            np.zeros((2, 1, 2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        )
        self.global_model = nn.Linear(4, 2)
        self.virtual_time = 0.0
        self.timeout = 1.0
        self.next_round_index = 0
        self.rng = random.Random(0)

    def run_rounds(self, count, *, record_default_metrics=False):
        return [
            self.run_one_round(record_default_metrics=record_default_metrics)
            for _ in range(count)
        ]

    def run_one_round(self, *, record_default_metrics=False):
        started = self.virtual_time
        self.virtual_time += 1.0
        self.next_round_index += 1
        return {
            "round": self.next_round_index,
            "started_time": started,
            "completed_time": self.virtual_time,
            "aggregated": True,
        }

    def evaluate_dataset(self, _dataset):
        return 0.25, 0.75


class _SmokeMonitor:
    def __init__(self) -> None:
        self.drift_events = []
        self.retrain_decisions = []
        self.counterfactual_triggers = []
        self.tick_history = []

    def observe_tick(self, batches_by_client, *, virtual_time, record_action=True, warmup=False):
        self.tick_history.append(
            {
                "time": virtual_time,
                "warmup": warmup,
                "clients": sorted(batches_by_client),
            }
        )
        return SimpleNamespace(should_retrain=False)


def _identity_corruption(batch, _kind, _severity, *, seed):
    return batch.clone()


class TestDriftWorkflowEndToEndSmoke(unittest.TestCase):
    def test_matrix_control_uses_identity_zero_and_persists_a_schema_v3_pair(self):
        config = DriftEpisodeConfig(
            dataset="e2e-smoke",
            num_clients=1,
            initial_rounds=1,
            retrain_rounds=1,
            warmup_ticks=20,
            monitor_tick_seconds=1.0,
            monitor_ticks=1,
            batch_size=1,
            corruption="gaussian_noise",
            severity=1,
            seed=17,
            production_horizon_seconds=1.0,
        )

        def runner(run_config, *, corruption_fn, **_kwargs):
            return run_drift_comparison(
                run_config,
                corruption_fn=corruption_fn,
                server_factory=_SmokeServer,
                monitor_factory=lambda _server, _baseline: _SmokeMonitor(),
            )

        with tempfile.TemporaryDirectory() as directory:
            paths = run_matrix(
                [config],
                output_dir=directory,
                runner=runner,
                corruption_fn=_identity_corruption,
                include_controls=True,
            )
            identity_pair = next(
                pair for pair in paths if pair[0].parent.parent.name == "identity_sev0"
            )

            agent, baseline = load_persisted_pair(identity_pair[0].parent)

            self.assertEqual(agent.metadata["corruption"], "identity")
            self.assertEqual(agent.metadata["severity"], 0)
            self.assertEqual(baseline.metadata["corruption"], "identity")
            self.assertEqual(baseline.metadata["severity"], 0)
            self.assertTrue((identity_pair[0].parent / "pair-manifest.json").is_file())

    def test_runs_persists_reloads_summarizes_and_plots_a_schema_v3_pair(self):
        config = DriftEpisodeConfig(
            dataset="e2e-smoke",
            num_clients=1,
            initial_rounds=1,
            retrain_rounds=1,
            warmup_ticks=20,
            monitor_tick_seconds=1.0,
            monitor_ticks=1,
            batch_size=1,
            corruption="gaussian_noise",
            severity=1,
            seed=17,
            production_horizon_seconds=1.0,
        )

        def runner(run_config, *, corruption_fn, **_kwargs):
            return run_drift_comparison(
                run_config,
                corruption_fn=corruption_fn,
                server_factory=_SmokeServer,
                monitor_factory=lambda _server, _baseline: _SmokeMonitor(),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = run_matrix(
                [config],
                output_dir=root,
                runner=runner,
                corruption_fn=_identity_corruption,
            )
            agent_path, baseline_path = paths[0]

            agent, baseline = load_persisted_pair(agent_path.parent)
            validate_pair(agent, baseline)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            image_path = plot_scenario(
                agent_path.parent.parent,
                root / "comparison.png",
            )

            self.assertEqual(agent.schema_version, 3)
            self.assertEqual(baseline.schema_version, 3)
            self.assertTrue((agent_path.parent / "pair-manifest.json").is_file())
            self.assertTrue(baseline_path.is_file())
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(len(summary["groups"]), 1)
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
