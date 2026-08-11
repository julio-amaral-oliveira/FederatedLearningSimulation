"""Pure helpers for the per-client drift schedule."""

import unittest

import numpy as np

from experiments.e07_drift_agent.drift_schedule import (
    build_mixed_dataset,
    build_retrain_datasets,
    corrupt_count,
    drifted_clients,
    is_client_drifted_at_tick,
    onset_tick,
    ramp_fraction,
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


class TestRampFraction(unittest.TestCase):
    def setUp(self):
        self.config = DriftEpisodeConfig(
            drift_ramp_ticks=20, monitor_tick_seconds=10.0
        )

    def test_fraction_is_zero_at_production_start(self):
        self.assertEqual(ramp_fraction(0.0, self.config), 0.0)

    def test_fraction_is_half_after_ten_ticks(self):
        self.assertEqual(ramp_fraction(100.0, self.config), 0.5)

    def test_fraction_clamps_at_one(self):
        self.assertEqual(ramp_fraction(300.0, self.config), 1.0)

    def test_fraction_is_one_without_ramp(self):
        config = DriftEpisodeConfig()
        self.assertEqual(ramp_fraction(0.0, config), 1.0)

    def test_fraction_clamps_negative_seconds_at_zero(self):
        self.assertEqual(ramp_fraction(-10.0, self.config), 0.0)

    def test_corrupt_count_rounds_the_fraction(self):
        self.assertEqual(corrupt_count(32, 0.0), 0)
        self.assertEqual(corrupt_count(32, 0.5), 16)
        self.assertEqual(corrupt_count(32, 1.0), 32)
        self.assertEqual(corrupt_count(4, 0.75), 3)

    def test_build_mixed_dataset_swaps_half_the_images_at_permutation_indices(self):
        clean = (
            np.zeros((8, 1, 1, 1), dtype=np.float32),
            np.arange(8, dtype=np.int64),
        )
        corrupted = (np.ones((8, 1, 1, 1), dtype=np.float32), clean[1].copy())
        permutation = np.array([3, 5, 0, 7, 2, 6, 1, 4])

        mixed = build_mixed_dataset(clean, corrupted, 0.5, permutation)

        self.assertTrue(np.array_equal(mixed[1], clean[1]))  # rótulos inalterados
        self.assertEqual(float(mixed[0].sum()), 4.0)  # exatamente metade corrompida
        self.assertTrue(np.all(mixed[0][permutation[:4]] == 1.0))
        kept = np.setdiff1d(np.arange(8), permutation[:4])
        self.assertTrue(np.all(mixed[0][kept] == 0.0))


if __name__ == "__main__":
    unittest.main()
