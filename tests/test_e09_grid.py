"""Grid builder and smoke runner for the E09 gradual-drift experiment."""

import unittest
from collections import Counter
from pathlib import Path

from experiments.e09_drift_gradual.run_grid import DEFAULT_SCENARIOS, build_grid


class TestBuildGrid(unittest.TestCase):
    def test_build_grid_covers_scenarios_seeds_and_ramp_durations(self):
        configs = build_grid()
        self.assertEqual(len(configs), 4 * 5 * 3)
        self.assertEqual(
            {c.drift_ramp_ticks for c in configs}, {5, 10, 20}
        )
        self.assertEqual({c.seed for c in configs}, {42, 43, 44, 45, 46})
        self.assertEqual(
            {(c.corruption, c.severity) for c in configs},
            set(DEFAULT_SCENARIOS),
        )

    def test_build_grid_fixes_policy_constants(self):
        for config in build_grid():
            self.assertIsNone(config.drift_onset_ticks)
            self.assertIsNone(config.drifted_client_ids)
            self.assertEqual(config.retrain_rounds, 1)
            self.assertEqual(config.production_horizon_seconds, 400.0)

    def test_result_directories_disambiguate_by_ramp_duration(self):
        from experiments.e07_drift_agent.run_matrix import _result_directory

        configs = build_grid()
        duplicates = Counter((c.corruption, c.severity, c.seed) for c in configs)
        directories = {
            _result_directory(Path("/tmp/e09-grid"), config, duplicates)
            for config in configs
        }
        self.assertEqual(len(directories), 60)


if __name__ == "__main__":
    unittest.main()
