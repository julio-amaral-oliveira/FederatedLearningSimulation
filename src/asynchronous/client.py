# client.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from constants import *


class Client:
    def __init__(
        self,
        dataset,
        client_id,
        train_time_range,
        speed_tier_name="fast",
    ):
        self.local_model = None
        self.optimizer = None
        self.dataset = dataset  # tuple (x, y)
        self.client_id = client_id
        self.train_time_range = train_time_range
        self.speed_tier_name = speed_tier_name

    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.local_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

    def setup_client(self, model):
        from utils.models import get_device, get_model_weights, set_model_weights
        self.local_model = type(model)().to(get_device())
        set_model_weights(self.local_model, get_model_weights(model))
        self._create_optimizer()

    def reset_optimizer(self):
        self._create_optimizer()

    def perform_fit(self, base_weights, local_epochs, batch_size):
        """Executa um round de treinamento local (sem sleeps, sem threading).
        Retorna os pesos atualizados."""
        from utils.models import get_device, get_model_weights, set_model_weights
        device = get_device()
        set_model_weights(self.local_model, base_weights)

        x, y = self.dataset
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        for _ in range(local_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                self.optimizer.zero_grad()
                outputs = self.local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        return get_model_weights(self.local_model)
