import unittest
from experiments.shared.comparison_core import compute_downtime


class TestComputeDowntime(unittest.TestCase):
    def test_empty_list_returns_zero(self):
        result = compute_downtime([], acc_threshold=0.5, end_time=10.0)
        self.assertEqual(result, 0.0)

    def test_uses_left_endpoint_for_above_below_and_below_above_transitions(self):
        entries = [
            {"accuracy": 0.8, "loss": 0.3, "time": 0.0},
            {"accuracy": 0.4, "loss": 0.8, "time": 10.0},
            {"accuracy": 0.8, "loss": 0.3, "time": 20.0},
        ]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=30.0)

        self.assertEqual(result, 10.0)

    def test_single_below_threshold_entry_counts_until_end_time(self):
        entries = [{"accuracy": 0.4, "loss": 0.8, "time": 10.0}]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=25.0)

        self.assertEqual(result, 15.0)

    def test_below_measurement_counts_entire_following_interval(self):
        entries = [
            {"accuracy": 0.4, "loss": 0.8, "time": 0.0},
            {"accuracy": 0.8, "loss": 0.3, "time": 7.0},
        ]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=7.0)

        self.assertEqual(result, 7.0)

    def test_unsorted_entries_are_ordered_before_computing_downtime(self):
        entries = [
            {"accuracy": 0.8, "loss": 0.3, "time": 7.0},
            {"accuracy": 0.4, "loss": 0.8, "time": 0.0},
        ]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=7.0)

        self.assertEqual(result, 7.0)

    def test_accuracy_equal_to_threshold_is_available(self):
        entries = [{"accuracy": 0.5, "loss": 0.5, "time": 0.0}]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=10.0)

        self.assertEqual(result, 0.0)

    def test_equal_timestamps_are_accepted(self):
        entries = [
            {"accuracy": 0.8, "loss": 0.3, "time": 10.0},
            {"accuracy": 0.4, "loss": 0.8, "time": 10.0},
        ]

        result = compute_downtime(entries, acc_threshold=0.5, end_time=20.0)

        self.assertEqual(result, 10.0)

    def test_end_time_before_final_entry_raises_value_error(self):
        entries = [
            {"accuracy": 0.3, "loss": 1.2, "time": 10.0},
            {"accuracy": 0.4, "loss": 1.0, "time": 20.0},
        ]

        with self.assertRaises(ValueError):
            compute_downtime(entries, acc_threshold=0.5, end_time=15.0)

    def test_negative_end_time_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_downtime([], acc_threshold=0.5, end_time=-1.0)


if __name__ == "__main__":
    unittest.main()
