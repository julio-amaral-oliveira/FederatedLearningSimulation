import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "synchronous"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synchronous.server import Server


class StubClient:
    def __init__(self, client_id, train_time_range=(1.0, 1.0)):
        self.client_id = client_id
        self.train_time_range = train_time_range
        self.speed_tier_name = "stub"
        self.was_setup = False

    def setup_client(self, model):
        self.was_setup = True

    def perform_fit(self, round_start_weights, local_epochs, batch_size):
        return [weight.clone() for weight in round_start_weights]

    def get_dataset_size(self):
        return 1


def make_server(*, timeout=10.0):
    data = (
        np.zeros((2, 1, 2, 2), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )
    clients = [StubClient(1), StubClient(2)]
    server = Server(
        clients=clients,
        num_clients=len(clients),
        num_rounds=5,
        timeout=timeout,
        local_epochs=1,
        batch_size=2,
        testing_data=data,
        model_name="cnn_mnist",
    )
    server.global_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    server.setup_clients()
    return server, data


class TestSynchronousServerIncrementalRounds(unittest.TestCase):
    def test_incremental_rounds_are_sequential_and_increase_time(self):
        server, _ = make_server()

        events = server.run_rounds(2) + server.run_rounds(3)

        self.assertEqual([event["round"] for event in events], [1, 2, 3, 4, 5])
        self.assertTrue(all(event["aggregated"] for event in events))
        self.assertEqual(events[0]["started_time"], 0.0)
        self.assertTrue(
            all(
                earlier["completed_time"] < later["completed_time"]
                for earlier, later in zip(events, events[1:])
            )
        )

    def test_seeded_incremental_runs_match_single_batch(self):
        first_server, _ = make_server()
        second_server, _ = make_server()

        single_batch = first_server.run_rounds(5)
        split_batch = second_server.run_rounds(2) + second_server.run_rounds(3)

        self.assertEqual(
            [event["completed_time"] for event in single_batch],
            [event["completed_time"] for event in split_batch],
        )

    def test_evaluate_dataset_restores_testing_data_and_train_mode(self):
        server, original_data = make_server()
        other_data = (
            np.ones((2, 1, 2, 2), dtype=np.float32),
            np.array([1, 0], dtype=np.int64),
        )
        server.global_model.train()

        loss, accuracy = server.evaluate_dataset(other_data)

        self.assertIs(server.testing_data, original_data)
        self.assertTrue(server.global_model.training)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)

    def test_run_rounds_rejects_zero(self):
        server, _ = make_server()

        with self.assertRaises(ValueError):
            server.run_rounds(0)


if __name__ == "__main__":
    unittest.main()
