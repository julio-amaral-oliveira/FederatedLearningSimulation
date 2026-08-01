"""Operational registry for the experiment families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe one experiment family and its official results root."""

    experiment_id: str
    title: str
    status: str
    output_root: Path


_EXPERIMENTS = (
    ExperimentSpec(
        "e01-static",
        "Treinamento federado estático",
        "historical",
        Path("results/e01-static"),
    ),
    ExperimentSpec(
        "e02-ablation",
        "Ablação da agregação assíncrona",
        "historical",
        Path("results/e02-ablation"),
    ),
    ExperimentSpec(
        "e03-timeout",
        "Impacto do timeout síncrono",
        "historical",
        Path("results/e03-timeout"),
    ),
    ExperimentSpec(
        "e04-static-comparison",
        "Comparação Sync contra Async em cenário estático",
        "historical",
        Path("results/e04-static-comparison"),
    ),
    ExperimentSpec(
        "e05-temporal-drift",
        "Drift temporal sazonal Sync contra Async",
        "historical",
        Path("results/e05-temporal-drift"),
    ),
    ExperimentSpec(
        "e06-drift-agent-prototype",
        "Protótipo inicial do drift agent",
        "legacy",
        Path("results/e06-drift-agent-prototype"),
    ),
    ExperimentSpec(
        "e07-drift-agent",
        "Drift agent endurecido e auditável",
        "current",
        Path("results/e07-drift-agent"),
    ),
)

EXPERIMENTS: tuple[ExperimentSpec, ...] = _EXPERIMENTS
_BY_ID = {spec.experiment_id: spec for spec in EXPERIMENTS}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    """Return a registered experiment or raise ``ValueError``."""

    try:
        return _BY_ID[experiment_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown experiment id: {experiment_id!r}") from error


def output_path(experiment_id: str, dataset: str) -> Path:
    """Return the published results root for an experiment and dataset."""

    return _dataset_path(experiment_id, dataset, root_kind="published")


def temporary_output_path(experiment_id: str, dataset: str) -> Path:
    """Return the non-canonical temporary output root for an experiment."""

    return _dataset_path(experiment_id, dataset, root_kind="temporary")


def _dataset_path(experiment_id: str, dataset: str, *, root_kind: str) -> Path:
    """Build a dataset path while keeping the registry as the identity seam."""

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty directory name")

    dataset_path = Path(dataset)
    if dataset_path.name != dataset or dataset in {".", ".."}:
        raise ValueError("dataset must be a single directory name")

    experiment = get_experiment(experiment_id)
    if root_kind == "published":
        return experiment.output_root / dataset
    if root_kind == "temporary":
        return Path("output") / experiment.experiment_id / dataset
    raise ValueError(f"unknown output root kind: {root_kind!r}")
