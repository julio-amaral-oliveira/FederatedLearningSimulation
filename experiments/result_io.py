"""Atomic persistence for standalone and paired drift artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from experiments.drift_results import DriftResult


def _stage_json(path: Path, payload: Mapping[str, Any]) -> tuple[Path, bytes]:
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        serialized = json.dumps(payload, indent=2).encode("utf-8")
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        return temporary_path, serialized
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish a complete JSON object at ``path``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path, _ = _stage_json(destination, payload)
    try:
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def save_validated_pair(
    output_dir: str | Path,
    agent: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Publish a validated canonical pair and its completion manifest."""

    from experiments.drift_results import validate_pair

    validate_pair(agent, baseline)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    agent_path = directory / "agent.json"
    baseline_path = directory / "baseline.json"
    manifest_path = directory / "pair-manifest.json"

    staged_agent: Path | None = None
    staged_baseline: Path | None = None
    try:
        staged_agent, agent_bytes = _stage_json(agent_path, agent)
        staged_baseline, baseline_bytes = _stage_json(baseline_path, baseline)
        manifest = {
            "schema_version": 1,
            "generation_id": uuid.uuid4().hex,
            "artifacts": {
                "agent": {
                    "filename": agent_path.name,
                    "sha256": hashlib.sha256(agent_bytes).hexdigest(),
                },
                "baseline": {
                    "filename": baseline_path.name,
                    "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                },
            },
        }

        manifest_path.unlink(missing_ok=True)
        os.replace(staged_agent, agent_path)
        staged_agent = None
        os.replace(staged_baseline, baseline_path)
        staged_baseline = None
        atomic_write_json(manifest_path, manifest)
    finally:
        if staged_agent is not None:
            staged_agent.unlink(missing_ok=True)
        if staged_baseline is not None:
            staged_baseline.unlink(missing_ok=True)

    return agent_path, baseline_path


def load_persisted_pair(
    output_dir: str | Path,
) -> tuple["DriftResult", "DriftResult"]:
    """Load a complete pair only when its manifest and byte digests are valid."""

    from experiments.drift_results import DriftResult, validate_pair

    directory = Path(output_dir)
    manifest_path = directory / "pair-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing drift pair manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid drift pair manifest at {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("drift pair manifest root must be an object")
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError(
            f"unsupported drift pair manifest schema_version: {schema_version!r}"
        )
    generation_id = manifest.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("drift pair manifest generation_id must be a non-empty string")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("drift pair manifest artifacts must be an object")

    serialized: dict[str, bytes] = {}
    for arm, expected_filename in (
        ("agent", "agent.json"),
        ("baseline", "baseline.json"),
    ):
        artifact = artifacts.get(arm)
        if not isinstance(artifact, dict):
            raise ValueError(f"drift pair manifest missing {arm} artifact")
        if artifact.get("filename") != expected_filename:
            raise ValueError(
                f"drift pair manifest {arm} filename must be {expected_filename}"
            )
        expected_digest = artifact.get("sha256")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"drift pair manifest {arm} digest is invalid")
        artifact_path = directory / expected_filename
        try:
            artifact_bytes = artifact_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"drift pair artifact is missing or unreadable: {artifact_path}"
            ) from error
        actual_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"drift pair {arm} digest mismatch: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        serialized[arm] = artifact_bytes

    payloads: dict[str, dict[str, Any]] = {}
    for arm in ("agent", "baseline"):
        try:
            payload = json.loads(serialized[arm])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid persisted {arm} drift result JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"persisted {arm} drift result root must be an object")
        payloads[arm] = payload

    agent = DriftResult.from_payload(payloads["agent"])
    baseline = DriftResult.from_payload(payloads["baseline"])
    validate_pair(agent, baseline)
    return agent, baseline
