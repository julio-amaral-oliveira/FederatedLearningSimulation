import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Client:
    def __init__(self, dataset, client_id, train_time_range, speed_tier_name="fast"):
        self.local_model = None
        self.dataset = dataset  # tuple (x, y)
        self.client_id = client_id
        self.train_time_range = train_time_range
        self.speed_tier_name = speed_tier_name

    def setup_client(self, model):
        from utils.models import get_device, get_model_weights, set_model_weights

        self.local_model = type(model)().to(get_device())
        set_model_weights(self.local_model, get_model_weights(model))
        self.optimizer = torch.optim.Adam(
            self.local_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

    def perform_fit(self, round_start_weights, local_epochs, batch_size):
        from utils.models import get_model_weights, set_model_weights

        set_model_weights(self.local_model, round_start_weights)
        self._fit(local_epochs, batch_size)
        return get_model_weights(self.local_model)

    def _fit(self, local_epochs, batch_size):
        from utils.models import get_device

        device = get_device()
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

    def get_dataset_size(self):
        return len(self.dataset[0])

    def set_model_weights(self, weights):
        from utils.models import set_model_weights

        set_model_weights(self.local_model, weights)

    def get_model_weights(self):
        from utils.models import get_model_weights

        return get_model_weights(self.local_model)
