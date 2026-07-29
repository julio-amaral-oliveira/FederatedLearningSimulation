"""Validated loading and comparison for synchronous drift experiment results."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_COMMON_REQUIRED = (
    "metadata",
    "corrupted_accuracy_history",
    "clean_evaluations",
    "metrics",
)
_V3_REQUIRED = (
    "experiment_config",
    "detector_config",
    "runtime",
    "tick_history",
    "retrain_decisions",
    "counterfactual_triggers",
    "retrain_round_events",
)
_LIST_FIELDS = (
    "corrupted_accuracy_history",
    "clean_evaluations",
    "drift_events",
    "retrain_decisions",
    "counterfactual_triggers",
    "tick_history",
    "retrain_round_events",
)


@dataclass(frozen=True)
class DriftResult:
    """One validated schema-v2 or schema-v3 drift result payload."""

    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DriftResult":
        if not isinstance(payload, Mapping):
            raise TypeError("drift result payload must be a mapping")
        data = copy.deepcopy(dict(payload))
        schema_version = data.get("schema_version")
        if schema_version not in (2, 3):
            raise ValueError(
                f"unsupported drift result schema_version: {schema_version}"
            )

        required = _COMMON_REQUIRED + (_V3_REQUIRED if schema_version == 3 else ())
        for field in required:
            if field not in data:
                raise ValueError(
                    f"schema v{schema_version} drift result requires {field}"
                )

        for field in ("metadata", "metrics"):
            if not isinstance(data[field], dict):
                raise ValueError(f"drift result {field} must be an object")
        for field in _LIST_FIELDS:
            if field in data and not isinstance(data[field], list):
                raise ValueError(f"drift result {field} must be a list")

        if schema_version == 3:
            for field in ("experiment_config", "detector_config", "runtime"):
                if not isinstance(data[field], dict):
                    raise ValueError(f"schema v3 drift result {field} must be an object")
            _validate_v3_internal_consistency(data)

        return cls(payload=data)

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def metadata(self) -> dict[str, Any]:
        return self.payload["metadata"]

    @property
    def metrics(self) -> dict[str, Any]:
        return self.payload["metrics"]

    @property
    def corrupted_accuracy_history(self) -> list[dict[str, Any]]:
        return self.payload["corrupted_accuracy_history"]

    @property
    def clean_evaluations(self) -> list[dict[str, Any]]:
        return self.payload["clean_evaluations"]

    @property
    def tick_history(self) -> list[dict[str, Any]]:
        return self.payload.get("tick_history", [])

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.payload.get("experiment_config", {})

    @property
    def detector_config(self) -> dict[str, Any]:
        return self.payload.get("detector_config", {})

    @property
    def runtime(self) -> dict[str, Any]:
        return self.payload.get("runtime", {})

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def _validate_v3_internal_consistency(payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    config = payload["experiment_config"]
    for field in ("dataset", "corruption", "severity", "seed", "tau", "baseline"):
        if field not in metadata:
            raise ValueError(f"schema v3 metadata requires {field}")
        if field not in config:
            raise ValueError(f"schema v3 experiment_config requires {field}")
        if metadata[field] != config[field]:
            raise ValueError(
                f"schema v3 {field} differs between metadata and experiment_config"
            )

    for field in (
        "production_start_time",
        "end_time_seconds",
        "production_horizon_seconds",
    ):
        if field not in metadata:
            raise ValueError(f"schema v3 metadata requires {field}")

    checkpoint = payload["runtime"].get("clean_checkpoint_digest")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(
            "schema v3 runtime requires non-empty clean_checkpoint_digest"
        )


def load_drift_result(path: str | Path) -> DriftResult:
    """Load and validate a drift result without upgrading legacy semantics."""

    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid drift result JSON at {result_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("drift result JSON root must be an object")
    return DriftResult.from_payload(payload)


def _as_result(value: DriftResult | Mapping[str, Any]) -> DriftResult:
    if isinstance(value, DriftResult):
        return value
    return DriftResult.from_payload(value)


def _pair_value(result: DriftResult, field: str) -> Any:
    if field in result.metadata:
        return result.metadata[field]
    return result.experiment_config.get(field)


def _require_equal(
    label: str,
    agent_value: Any,
    baseline_value: Any,
) -> None:
    if agent_value != baseline_value:
        raise ValueError(
            f"drift pair {label} mismatch: "
            f"agent={agent_value!r}, baseline={baseline_value!r}"
        )


def _validate_arm_identity(agent: DriftResult, baseline: DriftResult) -> None:
    if _pair_value(agent, "baseline") is not False:
        raise ValueError("drift pair agent payload must have baseline=False")
    if _pair_value(baseline, "baseline") is not True:
        raise ValueError("drift pair baseline payload must have baseline=True")


def _trace_through(
    trace: list[dict[str, Any]], decision_time: float | None
) -> list[dict[str, Any]]:
    if decision_time is None:
        return trace
    return [
        entry
        for entry in trace
        if isinstance(entry.get("time"), (int, float))
        and float(entry["time"]) <= decision_time
    ]


def _first_event(result: DriftResult, field: str) -> dict[str, Any] | None:
    events = result.get(field, [])
    return events[0] if events else None


def _validate_v3_pair(agent: DriftResult, baseline: DriftResult) -> None:
    for label, field in (
        ("corruption", "corruption"),
        ("corruption severity", "severity"),
        ("seed", "seed"),
    ):
        _require_equal(label, _pair_value(agent, field), _pair_value(baseline, field))

    _require_equal(
        "checkpoint",
        agent.runtime["clean_checkpoint_digest"],
        baseline.runtime["clean_checkpoint_digest"],
    )
    _require_equal(
        "production start",
        agent.metadata["production_start_time"],
        baseline.metadata["production_start_time"],
    )
    _require_equal(
        "horizon",
        (
            agent.metadata["production_horizon_seconds"],
            agent.metadata["end_time_seconds"],
        ),
        (
            baseline.metadata["production_horizon_seconds"],
            baseline.metadata["end_time_seconds"],
        ),
    )

    agent_config = {
        key: value
        for key, value in agent.experiment_config.items()
        if key != "baseline"
    }
    baseline_config = {
        key: value
        for key, value in baseline.experiment_config.items()
        if key != "baseline"
    }
    _require_equal("experiment configuration", agent_config, baseline_config)
    _require_equal(
        "detector configuration",
        agent.detector_config,
        baseline.detector_config,
    )

    agent_decision = _first_event(agent, "retrain_decisions")
    baseline_trigger = _first_event(baseline, "counterfactual_triggers")
    decision_time = (
        float(agent_decision["time"])
        if agent_decision is not None and "time" in agent_decision
        else None
    )
    _require_equal(
        "trace",
        _trace_through(agent.tick_history, decision_time),
        _trace_through(baseline.tick_history, decision_time),
    )
    _require_equal("trigger", agent_decision, baseline_trigger)

    if agent.get("counterfactual_triggers", []):
        raise ValueError("drift pair agent cannot contain counterfactual triggers")
    if baseline.get("retrain_decisions", []):
        raise ValueError("drift pair baseline cannot contain retrain decisions")
    if baseline.get("retrain_round_events", []):
        raise ValueError("drift pair baseline cannot contain retraining rounds")

    configured_rounds = int(agent.experiment_config["retrain_rounds"])
    expected_rounds = configured_rounds if agent_decision is not None else 0
    actual_rounds = len(agent.get("retrain_round_events", []))
    if actual_rounds != expected_rounds:
        raise ValueError(
            "drift pair agent retraining rounds mismatch: "
            f"expected={expected_rounds}, actual={actual_rounds}"
        )


def _validate_v2_pair(agent: DriftResult, baseline: DriftResult) -> None:
    for label, field in (
        ("corruption", "corruption"),
        ("corruption severity", "severity"),
        ("seed", "seed"),
        ("production start", "production_start_time"),
        ("horizon", "end_time_seconds"),
    ):
        _require_equal(label, _pair_value(agent, field), _pair_value(baseline, field))


def validate_pair(
    agent: DriftResult | Mapping[str, Any],
    baseline: DriftResult | Mapping[str, Any],
) -> None:
    """Reject mismatched pairs; v2 remains explicitly non-auditable."""

    agent_result = _as_result(agent)
    baseline_result = _as_result(baseline)
    _require_equal(
        "schema version",
        agent_result.schema_version,
        baseline_result.schema_version,
    )
    _validate_arm_identity(agent_result, baseline_result)
    if agent_result.schema_version == 3:
        _validate_v3_pair(agent_result, baseline_result)
    else:
        _validate_v2_pair(agent_result, baseline_result)


def _finite_metric(result: DriftResult, *names: str) -> float | None:
    for name in names:
        value = result.metrics.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    return None


def _clean_retention(result: DriftResult) -> float | None:
    reported = _finite_metric(result, "clean_retention_delta")
    if reported is not None:
        return reported
    pre_drift = next(
        (
            entry
            for entry in result.clean_evaluations
            if entry.get("stage") == "pre_drift"
        ),
        None,
    )
    final = next(
        (
            entry
            for entry in reversed(result.clean_evaluations)
            if entry.get("stage") == "final"
        ),
        None,
    )
    if pre_drift is None or final is None:
        return None
    try:
        return float(final["accuracy"]) - float(pre_drift["accuracy"])
    except (KeyError, TypeError, ValueError):
        return None


def summarize_pair(
    agent: DriftResult | Mapping[str, Any],
    baseline: DriftResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the comparison metrics shared by analysis and visualization."""

    agent_result = _as_result(agent)
    baseline_result = _as_result(baseline)
    validate_pair(agent_result, baseline_result)
    agent_downtime = _finite_metric(agent_result, "downtime_seconds")
    baseline_downtime = _finite_metric(baseline_result, "downtime_seconds")
    avoided = (
        baseline_downtime - agent_downtime
        if agent_downtime is not None and baseline_downtime is not None
        else None
    )
    avoided_percent = (
        100.0 * avoided / baseline_downtime
        if avoided is not None and baseline_downtime not in (None, 0.0)
        else None
    )
    audited = agent_result.schema_version == 3
    return {
        "agent_downtime_seconds": agent_downtime,
        "baseline_downtime_seconds": baseline_downtime,
        "downtime_avoided_seconds": avoided,
        "downtime_avoided_percent": avoided_percent,
        "detection_delay_seconds": _finite_metric(
            agent_result, "detection_delay_seconds"
        ),
        "time_to_recovery_seconds": _finite_metric(
            agent_result,
            "time_to_recovery_seconds",
            "recovery_duration_seconds",
        ),
        "clean_retention_delta": _clean_retention(agent_result),
        "audited_pair": audited,
        "legacy_unverified_pair": not audited,
    }
