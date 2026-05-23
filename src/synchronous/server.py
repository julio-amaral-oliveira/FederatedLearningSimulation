import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from constants import (
    MAX_CONNECTION_TIME,
    MIN_CONNECTION_TIME,
    SIMULATION_SEED,
)
from utils.models import get_device, get_model, get_model_weights, set_model_weights


class Server:
    def __init__(
        self,
        clients,
        num_clients,
        num_rounds,
        timeout,
        local_epochs,
        batch_size,
        testing_data,
        model_name,
        evaluation_frequency=1,
    ):
        self.clients = clients
        self.total_clients = num_clients
        self.total_rounds = num_rounds
        self.start_time = 0
        self.virtual_time = 0.0
        self.timeout = timeout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.global_model = get_model(model_name)
        self.testing_data = testing_data  # tuple (x_test, y_test)
        self.accuracy_history = []
        self.rng = random.Random(SIMULATION_SEED)
        self.evaluation_frequency = max(1, evaluation_frequency)

    def setup_clients(self):
        for client in self.clients:
            client.setup_client(self.global_model)

    def aggregate_round(
        self,
        participating_client_weights,
        participating_client_sizes,
        round_index,
        should_record_metrics=True,
    ):
        if len(participating_client_weights) == 0:
            print("Nao houve resposta de nenhum cliente, o modelo nao foi modificado")
            return
        print(f"Gerando novo modelo global. Rodada {round_index + 1}.")
        total_participating_samples = sum(participating_client_sizes)
        weighted_weights = []
        for weight_index in range(len(participating_client_weights[0])):
            aggregated_weight = torch.zeros_like(
                participating_client_weights[0][weight_index]
            )
            for client_index in range(len(participating_client_weights)):
                client_sample_fraction = (
                    participating_client_sizes[client_index]
                    / total_participating_samples
                )
                aggregated_weight += (
                    participating_client_weights[client_index][weight_index]
                    * client_sample_fraction
                )
            weighted_weights.append(aggregated_weight)
        set_model_weights(self.global_model, weighted_weights)
        if should_record_metrics:
            loss, accuracy, time_stamp = self.evaluate()
            self.accuracy_history.append((loss, accuracy, time_stamp))

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
        return avg_loss, accuracy, self.virtual_time

    def _sample_round_duration(self, client):
        connection = self.rng.uniform(MIN_CONNECTION_TIME, MAX_CONNECTION_TIME)
        train = self.rng.uniform(*client.train_time_range)
        return connection + train

    def train_clients(self, round_index, round_start_weights):
        participating_client_weights = []
        participating_client_sizes = []
        sampled_client_durations = []

        for client in self.clients:
            client_virtual_duration = self._sample_round_duration(client)
            sampled_client_durations.append(client_virtual_duration)
            if client_virtual_duration <= self.timeout:
                print(
                    f"Tempo virtual do cliente {client.client_id} "
                    f"({client.speed_tier_name}): {client_virtual_duration:.2f}s"
                )
                participating_client_weights.append(
                    client.perform_fit(
                        round_start_weights, self.local_epochs, self.batch_size
                    )
                )
                participating_client_sizes.append(client.get_dataset_size())
            else:
                print(
                    f"Cliente {client.client_id} excedeu o tempo limite virtual "
                    f"na rodada {round_index} "
                    f"({client_virtual_duration:.2f}s > {self.timeout:.2f}s)."
                )

        effective_round_duration = min(
            self.timeout, max(sampled_client_durations, default=0.0)
        )
        print(
            f"Percentual de clientes na rodada {round_index + 1}: "
            f"{100 * len(participating_client_weights) / self.total_clients}%"
        )
        print(
            f"Tempo virtual da rodada {round_index + 1}: "
            f"{effective_round_duration:.2f}s"
        )
        return (
            participating_client_weights,
            participating_client_sizes,
            effective_round_duration,
        )

    def start_training(self):
        self.start_time = time.time()
        self.virtual_time = 0.0
        self.rng = random.Random(SIMULATION_SEED)
        for round_index in range(self.total_rounds):
            print(f"\nRodada {round_index + 1}")
            round_start_weights = get_model_weights(self.global_model)
            (
                participating_client_weights,
                participating_client_sizes,
                effective_round_duration,
            ) = self.train_clients(
                round_index,
                round_start_weights,
            )
            self.virtual_time += effective_round_duration
            should_record_metrics = (
                round_index == 0
                or (round_index + 1) % self.evaluation_frequency == 0
                or round_index == self.total_rounds - 1
            )
            self.aggregate_round(
                participating_client_weights,
                participating_client_sizes,
                round_index,
                should_record_metrics=should_record_metrics,
            )

        wall_clock = time.time() - self.start_time
        print("Treinamento concluido. Novo modelo global gerado.")
        loss, accuracy, _ = self.evaluate()
        if (
            not self.accuracy_history
            or self.accuracy_history[-1][2] != self.virtual_time
        ):
            self.accuracy_history.append((loss, accuracy, self.virtual_time))

        print("Treinamento federado sincrono (tempo virtual) concluido.")
        print(f"Perda final do modelo global: {loss:.4f}")
        print(f"Acuracia final do modelo global: {accuracy:.4f}")
        print(
            f"Tempo virtual total: {self.virtual_time:.1f}s | "
            f"Wall-clock real: {wall_clock:.1f}s"
        )
        return self.accuracy_history
