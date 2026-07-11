import math
import warnings
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADWIN:
    """Adaptive Windowing drift detector (Bifet & Gavalda, 2007).

    Maintains a sliding window and searches for a split into two subwindows
    whose means differ by more than an epsilon bound. When such a split is
    found, the oldest elements are dropped until the window is homogeneous
    again.
    """

    def __init__(self, delta: float = 0.002, min_variance: float = 1e-6):
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if min_variance < 0:
            raise ValueError("min_variance must be non-negative")
        self.delta = delta
        self.min_variance = min_variance
        self._window = deque()
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> tuple[bool, float]:
        """Feed one scalar and return (drift_detected, current_mean)."""
        self._window.append(float(value))
        self._sum += value
        self._sum_sq += value * value

        drift_detected = False
        while len(self._window) > 2 and self._drift_detected():
            removed = self._window.popleft()
            self._sum -= removed
            self._sum_sq -= removed * removed
            drift_detected = True

        return drift_detected, self._mean()

    def _mean(self) -> float:
        n = len(self._window)
        return self._sum / n if n else 0.0

    def _drift_detected(self) -> bool:
        n = len(self._window)
        if n < 20:
            return False

        variance = self._sum_sq / n - (self._sum / n) ** 2
        variance = max(variance, self.min_variance)
        delta_prime = self.delta / n

        prefix = [0.0] * (n + 1)
        for i, val in enumerate(self._window, start=1):
            prefix[i] = prefix[i - 1] + val

        total = prefix[n]
        for i in range(10, n - 9):
            n0 = i
            n1 = n - i
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            mean0 = prefix[i] / n0
            mean1 = (total - prefix[i]) / n1
            epsilon = math.sqrt(
                (2.0 / m) * variance * math.log(2.0 / delta_prime)
            )
            if abs(mean0 - mean1) > epsilon:
                return True
        return False


class ShannonEntropy:
    """Computes Shannon entropy from a probability tensor.

    The input is expected to contain valid probabilities (non-negative and
    summing to one along the chosen dimension). A small floor is applied to
    avoid numerical instability in ``log``.
    """

    def __call__(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        safe = torch.clamp(probs, min=1e-12)
        return -torch.sum(safe * torch.log(safe), dim=dim)


class UDDDetector:
    """Uncertainty-aware drift detector using MC Dropout and ADWIN.

    For every batch, the model is evaluated in ``eval()`` mode with only its
    dropout layers temporarily enabled for ``T`` stochastic forwards. Shannon entropy is computed for each stochastic
    forward pass and the mean entropy over the ``T`` passes is fed into an
    ADWIN instance. A drift alarm is raised when the entropy stream changes
    significantly, which corresponds to the model becoming unexpectedly
    uncertain about its predictions.

    The detector never mutates the caller-owned model: it infers the device
    from the model's parameters and restores the model's training mode after
    each update.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.002,
        T: int = 5,
        device: Optional[str] = None,
    ):
        if T < 1:
            raise ValueError("T must be at least 1")

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        if device is not None:
            requested = torch.device(device)
            if requested != model_device:
                warnings.warn(
                    f"UDDDetector ignoring requested device {device!r}; "
                    f"using model device {model_device!r}",
                    stacklevel=2,
                )

        self.model = model
        self.T = T
        self.device = model_device
        self.adwin = ADWIN(delta=alpha)
        self._entropy = ShannonEntropy()

    def update(self, x_batch: torch.Tensor) -> tuple[bool, float]:
        """Run MC Dropout inference and feed mean entropy to ADWIN.

        Returns:
            ``(drift_flag, score)`` where ``score`` is the mean Shannon
            entropy over the ``T`` stochastic forward passes for the batch.
        """
        module_modes = {
            module: module.training for module in self.model.modules()
        }
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        try:
            x = x_batch.to(self.device)

            entropy_sum = 0.0
            with torch.no_grad():
                for _ in range(self.T):
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=-1)
                    entropy_sum += float(self._entropy(probs).sum().item())

            score = entropy_sum / (self.T * x.shape[0])
            drift_detected, _ = self.adwin.update(score)
            return drift_detected, score
        finally:
            for module, was_training in module_modes.items():
                module.train(was_training)

