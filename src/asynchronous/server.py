# server.py

import heapq
import random
import time

import torch
import torch.nn as nn
from constants import (
    MAX_CONNECTION_TIME,
    MIN_CONNECTION_TIME,
    SIMULATION_SEED,
)
from torch.utils.data import DataLoader, TensorDataset

from utils.models import get_device, get_model, get_model_weights, set_model_weights


class Server:
    def __init__(
        self,
        clients,
        num_clients,
        num_updates,
        timeout,
        local_epochs,
        batch_size,
        testing_data,
        model_name,
        base_alpha,
        decay_of_base_alpha,
        tardiness_sensivity,
        evaluation_frequency=1,
    ):
        self.clients = clients
        self.number_of_clients = num_clients
        self.number_of_updates = num_updates
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = get_model(model_name)
        self.testing_data = testing_data  # tuple (x_test, y_test)
        self.accuracy_history = []
        self.start_time = 0
        self.version = 0
        self.base_alpha = base_alpha
        self.decay_of_base_alpha = decay_of_base_alpha
        self.tardiness_sensitivity = tardiness_sensivity
        self.evaluation_frequency = max(1, evaluation_frequency)

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def update_global_weights(self, global_weights, updated_weights, client_version):
        staleness = self.version - client_version
        agg_factor = self.get_aggregation_factor(staleness)
        for i in range(len(global_weights)):
            global_weights[i] = (
                global_weights[i] * (1 - agg_factor) + agg_factor * updated_weights[i]
            )

    def get_aggregation_factor(self, staleness):
        return (
            self.base_alpha
            * (self.decay_of_base_alpha**self.version)
            * (1 / (1 + self.tardiness_sensitivity * staleness))
        )

    def _sample_round_duration(self, client, rng):
        connection = rng.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train = rng.uniform(*client.train_time_range)
        return connection + train

    def evaluate(self):
        device = get_device()
        self.global_model.eval()
        x, y = self.testing_data
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.global_model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def start_training(self):
        """Simulacao por eventos discretos: cada update do cliente eh um
        evento agendado em tempo virtual. O heap garante processamento em
        ordem cronologica (sem threads, sem sleeps, sem lock)."""
        self.start_time = time.time()
        rng = random.Random(SIMULATION_SEED)

        pq = []
        seq = 0
        initial_weights = get_model_weights(self.global_model)
        for idx, client in enumerate(self.clients):
            duration = self._sample_round_duration(client, rng)
            heapq.heappush(pq, (duration, seq, idx, 0, initial_weights))
            seq += 1

        last_accuracy = 0.0
        last_loss = None
        final_virtual_time = 0.0

        while pq:
            finish_time, _, client_idx, base_version, base_weights = heapq.heappop(pq)
            client = self.clients[client_idx]

            if finish_time > self.timeout:
                final_virtual_time = self.timeout
                late_ids = [client.client_id]
                late_ids.extend(
                    self.clients[c_idx].client_id for _, _, c_idx, _, _ in pq
                )
                for c_idx in late_ids:
                    print(f"Cliente {c_idx} excedeu o tempo limite virtual.")
                break

            updated_weights = client.perform_fit(
                base_weights, self.local_epochs, self.batch_size
            )

            global_weights = get_model_weights(self.global_model)
            self.update_global_weights(global_weights, updated_weights, base_version)
            set_model_weights(self.global_model, global_weights)

            next_version = self.version + 1
            should_eval = (
                next_version == 1
                or next_version % self.evaluation_frequency == 0
            )

            if should_eval:
                loss, accuracy = self.evaluate()
                last_loss = loss
                last_accuracy = accuracy
            else:
                accuracy = last_accuracy
                loss = last_loss

            if should_eval:
                self.accuracy_history.append((loss, accuracy, finish_time))

            staleness = self.version - base_version
            self.version += 1
            client.completed_updates += 1

            if should_eval:
                print(
                    f"[t_virtual={finish_time:7.2f}s] Cliente {client.client_id} | "
                    f"{client.speed_tier_name} | base_v={base_version} | "
                    f"staleness={staleness} | acc={accuracy:.4f}"
                )

            final_virtual_time = finish_time

            if client.completed_updates < self.number_of_updates:
                next_duration = self._sample_round_duration(client, rng)
                heapq.heappush(
                    pq,
                    (
                        finish_time + next_duration,
                        seq,
                        client_idx,
                        self.version,
                        get_model_weights(self.global_model),
                    ),
                )
                seq += 1

        loss, accuracy = self.evaluate()
        if (
            not self.accuracy_history
            or self.accuracy_history[-1][2] != final_virtual_time
        ):
            self.accuracy_history.append((loss, accuracy, final_virtual_time))
        wall_clock = time.time() - self.start_time
        print("Treinamento federado assíncrono (DES) concluído.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acurácia final do modelo global: {accuracy:.4f}")
        print(
            f"Wall-clock real: {wall_clock:.1f}s | Total de agregacoes: {self.version}"
        )
        return self.accuracy_history
