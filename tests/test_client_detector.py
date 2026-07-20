import sys
import os

# async client.py uses "from constants import *" so we need the package dir on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "asynchronous"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.synchronous.client import Client as SyncClient
from src.asynchronous.client import Client as AsyncClient


class _TinyLogitsModel(nn.Module):
    """Minimal model that returns fixed logits so tests are deterministic."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims if present; otherwise use as-is for already-flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


class _MockDetector:
    """Detector that tracks calls and returns configurable results."""

    def __init__(self, drift_flag: bool = False, score: float = 0.5):
        self.drift_flag = drift_flag
        self.score = score
        self.update_count = 0
        self.last_x = None

    def update(self, x_batch: torch.Tensor) -> tuple[bool, float]:
        self.update_count += 1
        self.last_x = x_batch
        return self.drift_flag, self.score


class TestSyncClientDetector(unittest.TestCase):
    def setUp(self):
        # Small dummy dataset: 4 samples, 784 features, labels 0-3
        x = torch.randn(4, 784)
        y = torch.tensor([0, 1, 2, 3])
        self.dataset = (x.numpy(), y.numpy())
        self.client = SyncClient(
            dataset=self.dataset,
            client_id=0,
            train_time_range=(1.0, 2.0),
            speed_tier_name="fast",
        )
        self.model = _TinyLogitsModel(num_classes=10)
        with torch.no_grad():
            self.model.fc.weight.fill_(0.0)
            self.model.fc.bias.fill_(0.0)
        self.client.setup_client(self.model)

    def test_infer_with_uncertainty_returns_correct_types_with_detector(self):
        mock_detector = _MockDetector(drift_flag=True, score=0.42)
        self.client.detector = mock_detector

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=3)

        self.assertIsInstance(pred, (int, torch.Tensor))
        if isinstance(pred, torch.Tensor):
            self.assertEqual(pred.shape[0], 2)  # batch of 2
            self.assertEqual(pred[0].item(), 0)  # argmax of zero logits
        self.assertIsInstance(drift_flag, bool)
        self.assertEqual(drift_flag, True)
        self.assertEqual(score, 0.42)

    def test_infer_with_uncertainty_returns_no_drift_without_detector(self):
        self.client.detector = None

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=3)

        self.assertIsInstance(pred, (int, torch.Tensor))
        self.assertIsInstance(drift_flag, bool)
        self.assertEqual(drift_flag, False)
        self.assertIsNone(score)

    def test_infer_with_uncertainty_runs_mc_dropout_with_detector(self):
        mock_detector = _MockDetector(drift_flag=False, score=0.1)
        self.client.detector = mock_detector

        x_batch = torch.randn(2, 784)
        self.client.infer_with_uncertainty(x_batch, T=5)

        # Detector should have been called once
        self.assertEqual(mock_detector.update_count, 1)
        self.assertIsNotNone(mock_detector.last_x)

    def test_infer_with_uncertainty_single_forward_without_detector(self):
        self.client.detector = None

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=1)

        self.assertEqual(pred.shape[0], 2)
        self.assertTrue((pred == 0).all())
        self.assertFalse(drift_flag)
        self.assertIsNone(score)

    def test_infer_with_uncertainty_restores_model_mode(self):
        self.client.local_model.eval()
        self.client.infer_with_uncertainty(torch.randn(2, 784))
        self.assertFalse(self.client.local_model.training)

    def test_perform_fit_still_exists_and_has_correct_signature(self):
        import inspect

        sig = inspect.signature(self.client.perform_fit)
        params = list(sig.parameters.keys())
        # Bound method: self is not listed in signature
        self.assertEqual(params, ["round_start_weights", "local_epochs", "batch_size"])

    def test_perform_fit_runs_without_error(self):
        # Minimal call to ensure body is unchanged (doesn't raise)
        initial_weights = [p.clone() for p in self.client.local_model.parameters()]
        result = self.client.perform_fit(initial_weights, local_epochs=1, batch_size=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(initial_weights))


class TestAsyncClientDetector(unittest.TestCase):
    def setUp(self):
        x = torch.randn(4, 784)
        y = torch.tensor([0, 1, 2, 3])
        self.dataset = (x.numpy(), y.numpy())
        self.client = AsyncClient(
            dataset=self.dataset,
            client_id=0,
            train_time_range=(1.0, 2.0),
            speed_tier_name="fast",
        )
        self.model = _TinyLogitsModel(num_classes=10)
        with torch.no_grad():
            self.model.fc.weight.fill_(0.0)
            self.model.fc.bias.fill_(0.0)
        self.client.setup_client(self.model)

    def test_infer_with_uncertainty_returns_correct_types_with_detector(self):
        mock_detector = _MockDetector(drift_flag=True, score=0.77)
        self.client.detector = mock_detector

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=3)

        self.assertIsInstance(pred, (int, torch.Tensor))
        if isinstance(pred, torch.Tensor):
            self.assertEqual(pred.shape[0], 2)
            self.assertEqual(pred[0].item(), 0)
        self.assertIsInstance(drift_flag, bool)
        self.assertEqual(drift_flag, True)
        self.assertEqual(score, 0.77)

    def test_infer_with_uncertainty_returns_no_drift_without_detector(self):
        self.client.detector = None

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=3)

        self.assertIsInstance(pred, (int, torch.Tensor))
        self.assertIsInstance(drift_flag, bool)
        self.assertEqual(drift_flag, False)
        self.assertIsNone(score)

    def test_infer_with_uncertainty_runs_mc_dropout_with_detector(self):
        mock_detector = _MockDetector(drift_flag=False, score=0.1)
        self.client.detector = mock_detector

        x_batch = torch.randn(2, 784)
        self.client.infer_with_uncertainty(x_batch, T=5)

        self.assertEqual(mock_detector.update_count, 1)
        self.assertIsNotNone(mock_detector.last_x)

    def test_infer_with_uncertainty_single_forward_without_detector(self):
        self.client.detector = None

        x_batch = torch.randn(2, 784)
        pred, drift_flag, score = self.client.infer_with_uncertainty(x_batch, T=1)

        self.assertEqual(pred.shape[0], 2)
        self.assertTrue((pred == 0).all())
        self.assertFalse(drift_flag)
        self.assertIsNone(score)

    def test_infer_with_uncertainty_restores_model_mode(self):
        self.client.local_model.eval()
        self.client.infer_with_uncertainty(torch.randn(2, 784))
        self.assertFalse(self.client.local_model.training)

    def test_perform_fit_still_exists_and_has_correct_signature(self):
        import inspect

        sig = inspect.signature(self.client.perform_fit)
        params = list(sig.parameters.keys())
        # Bound method: self is not listed in signature
        self.assertEqual(params, ["base_weights", "local_epochs", "batch_size"])

    def test_perform_fit_runs_without_error(self):
        initial_weights = [p.clone() for p in self.client.local_model.parameters()]
        result = self.client.perform_fit(initial_weights, local_epochs=1, batch_size=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(initial_weights))


if __name__ == "__main__":
    unittest.main()
