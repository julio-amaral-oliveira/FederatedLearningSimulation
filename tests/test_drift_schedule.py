"""Pure helpers for the per-client drift schedule."""

import unittest

import numpy as np

from experiments.e07_drift_agent.drift_schedule import (
    build_retrain_datasets,
    drifted_clients,
    is_client_drifted_at_tick,
    onset_tick,
)
from experiments.e07_drift_agent.episode import DriftEpisodeConfig


def _config(**overrides) -> DriftEpisodeConfig:
    defaults = dict(
        num_clients=10,
        drifted_client_ids=(0, 1, 2, 3, 4),
        drift_onset_ticks={0: 1, 1: 4, 2: 7, 3: 10, 4: 13},
    )
    defaults.update(overrides)
    return DriftEpisodeConfig(**defaults)


class TestDriftSchedule(unittest.TestCase):
    def test_default_schedule_means_all_clients_at_tick_zero(self):
        config = DriftEpisodeConfig(num_clients=10)
        self.assertEqual(drifted_clients(config), tuple(range(10)))
        self.assertEqual(onset_tick(3, config), 0)
        self.assertTrue(is_client_drifted_at_tick(3, 0, config))
        self.assertTrue(is_client_drifted_at_tick(3, 40, config))

    def test_partial_schedule_limits_drifted_clients(self):
        config = _config()
        self.assertEqual(drifted_clients(config), (0, 1, 2, 3, 4))
        self.assertFalse(is_client_drifted_at_tick(5, 40, config))
        self.assertFalse(is_client_drifted_at_tick(0, 0, config))
        self.assertTrue(is_client_drifted_at_tick(0, 1, config))
        self.assertFalse(is_client_drifted_at_tick(1, 3, config))
        self.assertTrue(is_client_drifted_at_tick(1, 4, config))

    def test_retrain_dataset_selects_current_distribution_per_client(self):
        clean = (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
        corrupted = (np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
        config = _config()
        datasets = build_retrain_datasets(
            [clean, clean, clean],
            [corrupted, corrupted, corrupted],
            current_tick=5,
            config=config,
        )
        self.assertTrue(np.array_equal(datasets[0][0], np.ones(4)))   # client 0, onset 1
        self.assertTrue(np.array_equal(datasets[1][0], np.ones(4)))   # client 1, onset 4
        self.assertTrue(np.array_equal(datasets[2][0], np.zeros(4)))  # client 2, onset 7


if __name__ == "__main__":
    unittest.main()
