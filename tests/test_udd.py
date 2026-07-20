import unittest
from typing import Optional

import torch
import torch.nn as nn

from src.utils.drift_detector import UDDDetector


class _MockClassifier(nn.Module):
    """Deterministic mock whose logits are either peaked or uniform."""

    def __init__(
        self, num_classes: int = 10, switch_after: Optional[int] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.switch_after = switch_after
        self._calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        batch_size = x.shape[0]
        peaked = (
            self.switch_after is None or self._calls <= self.switch_after
        )
        if peaked:
            logits = torch.full(
                (batch_size, self.num_classes), -10.0, dtype=torch.float32
            )
            logits[:, 0] = 10.0
            return logits
        return torch.zeros(batch_size, self.num_classes, dtype=torch.float32)


class TestUDDDetector(unittest.TestCase):
    def test_flags_uniform_logits_as_drift_after_certain_baseline(self):
        """Given a baseline of certain predictions then uncertain ones,
        UDDDetector must raise a drift alarm.
        """
        # Given: a mock that is certain for 20 batches, then uncertain
        model = _MockClassifier(num_classes=10, switch_after=20)
        detector = UDDDetector(model, alpha=0.002, T=5, device="cpu")

        # When: feeding the model until it switches to uniform logits
        detected = False
        for _ in range(100):
            batch = torch.randn(8, 3, 32, 32)
            drift_detected, _ = detector.update(batch)
            if drift_detected:
                detected = True
                break

        # Then: drift is reported after the onset of uncertainty
        self.assertTrue(detected)

    def test_stays_quiet_when_predictions_remain_peaked(self):
        """Given a stream of certain predictions, UDDDetector must stay quiet."""
        # Given: a mock that always returns peaked logits
        model = _MockClassifier(num_classes=10)
        detector = UDDDetector(model, alpha=0.002, T=5, device="cpu")

        # When: feeding many batches of peaked predictions
        for _ in range(100):
            batch = torch.randn(8, 3, 32, 32)
            drift_detected, _ = detector.update(batch)

            # Then: no drift is reported at any point
            self.assertFalse(drift_detected)


class TestUDDDetectorModelOwnership(unittest.TestCase):
    def test_does_not_move_model_to_requested_device(self):
        """UDDDetector must infer the model device and never move the model."""
        model = nn.Linear(4, 2)
        original_device = next(model.parameters()).device

        detector = UDDDetector(model, alpha=0.002, T=5, device="cuda")

        self.assertEqual(next(model.parameters()).device, original_device)
        self.assertEqual(detector.device, original_device)

    def test_restores_training_mode_after_update(self):
        """UDDDetector must restore the model's training mode after update."""
        model = nn.Linear(4, 2)
        model.eval()
        detector = UDDDetector(model, alpha=0.002, T=5)

        detector.update(torch.randn(2, 4))

        self.assertFalse(model.training)

    def test_preserves_every_module_mode_and_batch_norm_statistics(self):
        class _BatchNormDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_norm = nn.BatchNorm1d(4)
                self.dropout = nn.Dropout(p=0.5)
                self.fc = nn.Linear(4, 2)
                self.batch_norm_modes_during_forward = []
                self.dropout_modes_during_forward = []

            def forward(self, x):
                self.batch_norm_modes_during_forward.append(self.batch_norm.training)
                self.dropout_modes_during_forward.append(self.dropout.training)
                return self.fc(self.dropout(self.batch_norm(x)))

        model = _BatchNormDropoutModel()
        model.train()
        original_modes = {module: module.training for module in model.modules()}
        original_running_mean = model.batch_norm.running_mean.clone()
        detector = UDDDetector(model, T=3)

        detector.update(torch.randn(8, 4))

        self.assertEqual(
            {module: module.training for module in model.modules()}, original_modes
        )
        self.assertEqual(model.batch_norm_modes_during_forward, [False] * 3)
        self.assertEqual(model.dropout_modes_during_forward, [True] * 3)
        self.assertTrue(torch.equal(model.batch_norm.running_mean, original_running_mean))


if __name__ == "__main__":
    unittest.main()
