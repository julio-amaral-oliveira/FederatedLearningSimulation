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
    "drift_events",
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
_TIME_REL_TOL = 1e-9
_TIME_ABS_TOL = 1e-9


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
        if type(schema_version) is not int or schema_version not in (2, 3):
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
    detector = payload["detector_config"]
    runtime = payload["runtime"]
    for field in (
        "dataset",
        "corruption",
        "severity",
        "seed",
        "tau",
        "baseline",
        "production_horizon_seconds",
    ):
        if field not in metadata:
            raise ValueError(f"schema v3 metadata requires {field}")
        if field not in config:
            raise ValueError(f"schema v3 experiment_config requires {field}")
        if metadata[field] != config[field]:
            raise ValueError(
                f"schema v3 {field} differs between metadata and experiment_config"
            )

    corruption = metadata["corruption"]
    if not isinstance(corruption, str) or not corruption.strip():
        raise ValueError("schema v3 corruption must be a non-empty string")
    severity = metadata["severity"]
    if corruption == "identity":
        if not _is_integer(severity) or severity != 0:
            raise ValueError("schema v3 identity severity must be integer 0")
    elif not _is_integer(severity) or not 1 <= severity <= 5:
        raise ValueError(
            "schema v3 non-identity severity must be an integer in [1, 5]"
        )
    seed = metadata["seed"]
    if not _is_integer(seed):
        raise ValueError("schema v3 seed must be an integer")
    tau = metadata["tau"]
    if not _is_finite_number(tau) or not 0.0 <= float(tau) <= 1.0:
        raise ValueError("schema v3 tau must be numeric in [0, 1]")
    if not isinstance(metadata["baseline"], bool):
        raise ValueError("schema v3 baseline must be a bool")

    for field in ("production_start_time", "end_time_seconds"):
        if field not in metadata:
            raise ValueError(f"schema v3 metadata requires {field}")
        if not _is_finite_number(metadata[field]):
            raise ValueError(f"schema v3 {field} must be finite")
    production_start = float(metadata["production_start_time"])
    end_time = float(metadata["end_time_seconds"])
    horizon = metadata["production_horizon_seconds"]
    if not _is_finite_number(horizon) or float(horizon) <= 0.0:
        raise ValueError(
            "schema v3 production_horizon_seconds must be a positive finite number"
        )
    if end_time < production_start:
        raise ValueError(
            "schema v3 end_time_seconds must not precede production_start_time"
        )
    expected_end = production_start + float(horizon)
    if not math.isclose(
        end_time,
        expected_end,
        rel_tol=_TIME_REL_TOL,
        abs_tol=_TIME_ABS_TOL,
    ):
        raise ValueError(
            "schema v3 end_time_seconds must equal production_start_time + "
            "production_horizon_seconds"
        )

    if "retrain_rounds" not in config or not _is_integer(config["retrain_rounds"]):
        raise ValueError("schema v3 experiment_config requires integer retrain_rounds")
    if config["retrain_rounds"] < 1:
        raise ValueError("schema v3 retrain_rounds must be at least 1")

    ramp = config.get("drift_ramp_ticks")
    if ramp is not None and (
        not isinstance(ramp, int) or isinstance(ramp, bool) or ramp < 1
    ):
        raise ValueError(
            "schema v3 drift_ramp_ticks must be None or an integer >= 1"
        )
    if ramp is not None and (
        config.get("drift_onset_ticks") is not None
        or config.get("drifted_client_ids") is not None
    ):
        raise ValueError(
            "schema v3 drift_ramp_ticks cannot be combined with "
            "drift_onset_ticks or drifted_client_ids"
        )

    _validate_detector_config(detector)
    _validate_runtime(runtime)
    _validate_v3_histories(payload, production_start, end_time)
    _validate_v3_training_history(payload, production_start)
    _validate_v3_actions(payload, production_start, end_time)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_detector_config(detector: dict[str, Any]) -> None:
    required = (
        "detector_kind",
        "detector_alpha",
        "detector_T",
        "trigger_threshold",
        "trigger_window_ticks",
    )
    for field in required:
        if field not in detector:
            raise ValueError(f"schema v3 detector_config requires {field}")

    if detector["detector_kind"] != "udd":
        raise ValueError("schema v3 detector_kind must be 'udd'")
    alpha = detector["detector_alpha"]
    if not _is_finite_number(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("schema v3 detector_alpha must be numeric in (0, 1)")
    mc_passes = detector["detector_T"]
    if not _is_integer(mc_passes) or mc_passes < 1:
        raise ValueError("schema v3 detector_T must be an integer >= 1")
    threshold = detector["trigger_threshold"]
    if (
        not _is_finite_number(threshold)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("schema v3 trigger_threshold must be numeric in [0, 1]")
    window_ticks = detector["trigger_window_ticks"]
    if not _is_integer(window_ticks) or window_ticks < 1:
        raise ValueError("schema v3 trigger_window_ticks must be an integer >= 1")


def _validate_runtime(runtime: dict[str, Any]) -> None:
    for field in (
        "python_version",
        "numpy_version",
        "torch_version",
        "effective_device",
        "clean_checkpoint_digest",
    ):
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"schema v3 runtime requires non-empty {field}")


def _timestamp(
    entry: Any,
    *,
    field: str,
    key: str = "time",
) -> float:
    if not isinstance(entry, dict):
        raise ValueError(f"schema v3 {field} entries must be objects")
    if key not in entry or not _is_finite_number(entry[key]):
        raise ValueError(f"schema v3 {field} {key} must be finite")
    return float(entry[key])


def _within_interval(value: float, start: float, end: float) -> bool:
    return start <= value <= end


def _validate_timestamped_history(
    entries: list[Any],
    *,
    field: str,
    start: float,
    end: float,
    allow_warmup: bool = False,
) -> None:
    previous: float | None = None
    for entry in entries:
        timestamp = _timestamp(entry, field=field)
        if previous is not None and timestamp < previous:
            raise ValueError(
                f"schema v3 {field} timestamps must be monotonically non-decreasing"
            )
        if allow_warmup and entry.get("warmup") is True:
            if timestamp > start:
                raise ValueError(
                    f"schema v3 {field} warmup timestamps must not follow production start"
                )
        elif not _within_interval(timestamp, start, end):
            raise ValueError(
                f"schema v3 {field} timestamps must remain within the production interval"
            )
        previous = timestamp


def _validate_v3_histories(
    payload: dict[str, Any],
    production_start: float,
    end_time: float,
) -> None:
    for field in (
        "corrupted_accuracy_history",
        "clean_evaluations",
        "drift_events",
        "retrain_decisions",
        "counterfactual_triggers",
    ):
        _validate_timestamped_history(
            payload[field],
            field=field,
            start=production_start,
            end=end_time,
        )
    _validate_timestamped_history(
        payload["tick_history"],
        field="tick_history",
        start=production_start,
        end=end_time,
        allow_warmup=True,
    )


def _validate_v3_training_history(
    payload: dict[str, Any],
    production_start: float,
) -> None:
    """Validate the pre-production clean training history when present.

    The field is optional so published schema-v3 artifacts without it stay
    loadable.  When present, every timestamp must be finite, ordered, and
    precede production start.
    """
    entries = payload.get("clean_training_history", [])
    if not isinstance(entries, list):
        raise ValueError("schema v3 clean_training_history must be a list")
    previous: float | None = None
    for index, entry in enumerate(entries):
        timestamp = _timestamp(entry, field=f"clean_training_history[{index}]")
        if previous is not None and timestamp < previous:
            raise ValueError(
                "schema v3 clean_training_history timestamps must be "
                "monotonically non-decreasing"
            )
        if timestamp > production_start:
            raise ValueError(
                "schema v3 clean_training_history timestamps must not follow "
                "production start"
            )
        previous = timestamp


def _validate_v3_actions(
    payload: dict[str, Any],
    production_start: float,
    end_time: float,
) -> None:
    baseline = payload["metadata"]["baseline"]
    decisions = payload["retrain_decisions"]
    counterfactuals = payload["counterfactual_triggers"]
    rounds = payload["retrain_round_events"]

    if baseline:
        if decisions:
            raise ValueError("schema v3 baseline must have zero retrain_decisions")
        if rounds:
            raise ValueError("schema v3 baseline must have zero retrain_round_events")
        if len(counterfactuals) > 1:
            raise ValueError(
                "schema v3 baseline must have at most one counterfactual trigger"
            )
        return

    if counterfactuals:
        raise ValueError("schema v3 agent cannot contain counterfactual triggers")
    if len(decisions) > 1:
        raise ValueError("schema v3 agent must have at most one retrain decision")

    skipped = payload.get("metrics", {}).get("retrain_skipped_budget")
    if skipped is not None and not isinstance(skipped, bool):
        raise ValueError("schema v3 retrain_skipped_budget must be a bool")
    if skipped is True and not decisions:
        raise ValueError(
            "schema v3 retrain_skipped_budget requires a retrain decision"
        )

    expected_rounds = (
        payload["experiment_config"]["retrain_rounds"] if decisions else 0
    )
    if skipped is True:
        expected_rounds = 0
    if len(rounds) != expected_rounds:
        raise ValueError(
            "schema v3 retrain_round_events count mismatch: "
            f"expected={expected_rounds}, actual={len(rounds)}"
        )
    if not decisions:
        return

    decision_time = _timestamp(decisions[0], field="retrain_decisions")
    previous_completed = decision_time
    previous_round: int | None = None
    for event in rounds:
        started = _timestamp(
            event,
            field="retrain_round_events",
            key="started_time",
        )
        completed = _timestamp(
            event,
            field="retrain_round_events",
            key="completed_time",
        )
        round_number = event.get("round") if isinstance(event, dict) else None
        if not _is_integer(round_number):
            raise ValueError(
                "schema v3 retrain_round_events round must be an integer"
            )
        if previous_round is not None and round_number <= previous_round:
            raise ValueError(
                "schema v3 retrain_round_events rounds must be strictly ordered"
            )
        if started < previous_completed:
            raise ValueError(
                "schema v3 retrain_round_events must be ordered after the decision"
            )
        if completed < started:
            raise ValueError(
                "schema v3 retrain_round_events completed_time must not precede "
                "started_time"
            )
        if (
            not _within_interval(started, production_start, end_time)
            or not _within_interval(completed, production_start, end_time)
        ):
            raise ValueError(
                "schema v3 retrain_round_events must remain within the fixed horizon"
            )
        previous_completed = completed
        previous_round = round_number


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
    _require_equal(
        "training history",
        agent.get("clean_training_history", []),
        baseline.get("clean_training_history", []),
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
    if agent.get("metrics", {}).get("retrain_skipped_budget") is True:
        expected_rounds = 0
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
