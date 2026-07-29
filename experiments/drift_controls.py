"""Deterministic controls used to calibrate drift experiment behavior."""

from __future__ import annotations

import torch

from src.orchestrator.orchestrator import TickOutcome


def identity_corruption(
    inputs: torch.Tensor,
    _corruption: str = "identity",
    _severity: int = 0,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Return an independent, unchanged tensor for a no-drift control."""
    del seed
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("identity_corruption expects a torch.Tensor")
    return inputs.detach().clone()


class OracleMonitor:
    """A monitor that deterministically triggers once at production onset.

    It implements the same narrow monitor protocol consumed by
    :func:`experiments.smoke_drift.run_drift_episode`, without consulting
    labels or the model.  Warm-up ticks are recorded but never trigger.
    """

    def __init__(self) -> None:
        self._triggered = False
        self.drift_events: list[dict[str, float | str]] = []
        self.retrain_decisions: list[dict[str, float]] = []
        self.counterfactual_triggers: list[dict[str, float]] = []
        self.tick_history: list[dict] = []

    def observe_tick(
        self,
        batches_by_client: dict[str, torch.Tensor],
        *,
        virtual_time: float,
        record_action: bool = True,
        warmup: bool = False,
    ) -> TickOutcome:
        """Trigger exactly once on the first tick after clean warm-up."""
        for batch in batches_by_client.values():
            if not isinstance(batch, torch.Tensor):
                raise TypeError("batches_by_client values must be torch.Tensor")

        if warmup:
            self.tick_history.append(
                {
                    "time": virtual_time,
                    "warmup": True,
                    "clients": {},
                    "flagged_fraction": 0.0,
                }
            )
            return TickOutcome(False, 0.0, virtual_time)

        should_retrain = not self._triggered
        flagged_fraction = 1.0 if should_retrain else 0.0
        self.tick_history.append(
            {
                "time": virtual_time,
                "warmup": False,
                "clients": {},
                "flagged_fraction": flagged_fraction,
            }
        )
        if should_retrain:
            self._triggered = True
            self.drift_events.append(
                {"time": virtual_time, "client_id": "oracle", "score": 1.0}
            )
            decision = {"time": virtual_time, "flagged_fraction": 1.0}
            if record_action:
                self.retrain_decisions.append(decision)
            else:
                self.counterfactual_triggers.append(decision)
        return TickOutcome(should_retrain, flagged_fraction, virtual_time)
