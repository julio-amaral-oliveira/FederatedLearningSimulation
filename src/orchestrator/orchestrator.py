"""Tick-based, UDD-only drift monitoring.

The monitor observes one unlabeled batch for each client in a virtual-time
tick.  It records drift evidence but deliberately has no dependency on a
server and never starts retraining itself.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn

from src.utils.drift_detector import UDDDetector


class _Detector(Protocol):
    def update(self, x_batch: torch.Tensor) -> tuple[bool, float]:
        ...


@dataclass(frozen=True)
class TickOutcome:
    """The retraining decision resulting from one virtual-time tick."""

    should_retrain: bool
    flagged_fraction: float
    time: float


class DriftMonitor:
    """Aggregate per-client UDD alarms over a sliding window of ticks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        alpha: float = 0.002,
        T: int = 5,
        window_ticks: int = 50,
        trigger_threshold: float = 0.30,
        detectors: Optional[dict[str, _Detector]] = None,
    ):
        if window_ticks < 1:
            raise ValueError("window_ticks must be at least 1")
        if not 0.0 <= trigger_threshold <= 1.0:
            raise ValueError("trigger_threshold must be in [0, 1]")

        self._model = model
        self._alpha = alpha
        self._T = T
        self.window_ticks = window_ticks
        self.trigger_threshold = trigger_threshold
        self._detectors: dict[str, _Detector] = detectors or {}
        self._flag_maps: deque[dict[str, bool]] = deque(maxlen=window_ticks)
        self._action_latched = False
        self.drift_events: list[dict[str, float | str]] = []
        self.retrain_decisions: list[dict[str, float]] = []
        self.counterfactual_triggers: list[dict[str, float]] = []
        self.tick_history: list[dict] = []

    def _get_or_create_detector(self, client_id: str) -> _Detector:
        if client_id not in self._detectors:
            self._detectors[client_id] = UDDDetector(
                self._model, alpha=self._alpha, T=self._T
            )
        return self._detectors[client_id]

    def observe_tick(
        self,
        batches_by_client: dict[str, torch.Tensor],
        *,
        virtual_time: float,
        record_action: bool = True,
        warmup: bool = False,
    ) -> TickOutcome:
        """Observe exactly one unlabeled batch per client for one tick.

        ``warmup`` updates the local detectors without contributing evidence to
        the collective policy.  A detector can therefore calibrate on clean
        traffic without producing an externally visible drift event or
        consuming the monitor's one permitted retraining action.
        """
        flag_map: dict[str, bool] = {}
        clients: dict[str, dict[str, float | bool]] = {}
        for client_id, x_batch in batches_by_client.items():
            if not isinstance(x_batch, torch.Tensor):
                raise TypeError("batches_by_client values must be torch.Tensor")
            drift_flag, score = self._get_or_create_detector(client_id).update(
                x_batch
            )
            flag_map[client_id] = bool(drift_flag)
            clients[client_id] = {"score": float(score), "flag": bool(drift_flag)}
            if drift_flag and not warmup:
                self.drift_events.append(
                    {
                        "time": virtual_time,
                        "client_id": client_id,
                        "score": float(score),
                    }
                )

        if warmup:
            self.tick_history.append(
                {
                    "time": virtual_time,
                    "warmup": True,
                    "clients": clients,
                    "flagged_fraction": 0.0,
                }
            )
            return TickOutcome(
                should_retrain=False,
                flagged_fraction=0.0,
                time=virtual_time,
            )

        self._flag_maps.append(flag_map)
        flagged_clients = {
            client_id
            for tick_flags in self._flag_maps
            for client_id, flagged in tick_flags.items()
            if flagged
        }
        client_count = len(batches_by_client)
        flagged_fraction = len(flagged_clients) / client_count if client_count else 0.0
        self.tick_history.append(
            {
                "time": virtual_time,
                "warmup": False,
                "clients": clients,
                "flagged_fraction": flagged_fraction,
            }
        )
        trigger = flagged_fraction >= self.trigger_threshold
        should_retrain = trigger and not self._action_latched

        if should_retrain:
            decision = {"time": virtual_time, "flagged_fraction": flagged_fraction}
            if record_action:
                self.retrain_decisions.append(decision)
            else:
                self.counterfactual_triggers.append(decision)
            self._action_latched = True

        return TickOutcome(
            should_retrain=should_retrain,
            flagged_fraction=flagged_fraction,
            time=virtual_time,
        )
