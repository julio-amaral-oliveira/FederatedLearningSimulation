"""Plan or publish experiment output migrations."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experiments.shared.registry import ExperimentSpec, get_experiment


@dataclass(frozen=True)
class MigrationEntry:
    """Map one source file to its planned destination."""

    relative_path: Path
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class MigrationPlan:
    """Describe a migration without creating or changing any file."""

    experiment: ExperimentSpec
    source: Path
    destination: Path
    entries: tuple[MigrationEntry, ...]

    @property
    def file_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class CopyReport:
    """Summarize one validated publication."""

    destination: Path
    file_count: int
    pair_count: int


def build_plan(
    source: str | Path,
    destination: str | Path,
    experiment_id: str,
) -> MigrationPlan:
    """Build a plan from every regular file below ``source``."""

    experiment = get_experiment(experiment_id)
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"migration source directory does not exist: {source_path}"
        )

    source_resolved = source_path.resolve()
    destination_resolved = destination_path.resolve()
    if (
        destination_resolved == source_resolved
        or source_resolved in destination_resolved.parents
    ):
        raise ValueError("migration destination must not be inside the source directory")

    entries = []
    candidates = sorted(
        (
            path
            for path in source_path.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(source_path).as_posix(),
    )
    for path in candidates:
        relative_path = path.relative_to(source_path)
        entries.append(
            MigrationEntry(
                relative_path=relative_path,
                source_path=path,
                destination_path=destination_path / relative_path,
            )
        )

    return MigrationPlan(
        experiment=experiment,
        source=source_path,
        destination=destination_path,
        entries=tuple(entries),
    )


def render_dry_run(plan: MigrationPlan) -> str:
    """Render every planned source-to-destination mapping."""

    destination_state = "exists" if plan.destination.exists() else "absent"
    lines = [
        "DRY RUN: no files will be copied.",
        f"experiment: {plan.experiment.experiment_id}",
        f"source: {plan.source}",
        f"destination: {plan.destination}",
        f"destination_state: {destination_state}",
        f"file_count: {plan.file_count}",
    ]
    lines.extend(
        f"would_copy: {entry.source_path} -> {entry.destination_path}"
        for entry in plan.entries
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_directories(root: Path) -> tuple[Path, ...]:
    return tuple(sorted({path.parent for path in root.rglob("pair-manifest.json")}))


def _validate_pairs(root: Path) -> int:
    from experiments.shared.result_io import load_persisted_pair

    pair_directories = _pair_directories(root)
    for pair_directory in pair_directories:
        load_persisted_pair(pair_directory)
    return len(pair_directories)


def _copy_order(plan: MigrationPlan) -> tuple[MigrationEntry, ...]:
    return tuple(
        sorted(
            plan.entries,
            key=lambda entry: (
                entry.relative_path.name == "pair-manifest.json",
                entry.relative_path.as_posix(),
            ),
        )
    )


def _validate_staged_files(plan: MigrationPlan, staging: Path) -> None:
    expected_paths = {entry.relative_path for entry in plan.entries}
    actual_paths = {
        path.relative_to(staging)
        for path in staging.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "staged file set differs from source: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for entry in plan.entries:
        staged_path = staging / entry.relative_path
        if _sha256(entry.source_path) != _sha256(staged_path):
            raise ValueError(f"staged file digest mismatch: {entry.relative_path}")


def copy_plan(plan: MigrationPlan) -> CopyReport:
    """Copy and validate a plan without changing its source directory."""

    if plan.destination.exists():
        raise FileExistsError(
            f"migration destination already exists: {plan.destination}"
        )

    source_pair_count = _validate_pairs(plan.source)
    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.destination.name}.",
            dir=plan.destination.parent,
        )
    )
    published = False
    try:
        for entry in _copy_order(plan):
            staged_path = staging / entry.relative_path
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.source_path, staged_path)

        _validate_staged_files(plan, staging)
        staged_pair_count = _validate_pairs(staging)
        if staged_pair_count != source_pair_count:
            raise ValueError(
                "staged pair count differs from source: "
                f"source={source_pair_count}, staged={staged_pair_count}"
            )

        staging.replace(plan.destination)
        published = True
        return CopyReport(
            destination=plan.destination,
            file_count=plan.file_count,
            pair_count=staged_pair_count,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def render_copy_report(plan: MigrationPlan, report: CopyReport) -> str:
    """Render the result of a validated copy."""

    return "\n".join(
        (
            "COPY COMPLETE: source files were preserved.",
            f"experiment: {plan.experiment.experiment_id}",
            f"source: {plan.source}",
            f"destination: {report.destination}",
            f"file_count: {report.file_count}",
            f"pair_count: {report.pair_count}",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or publish an experiment output migration"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--experiment", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="list the planned mappings without copying files",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="copy, validate and publish the planned files",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args.source, args.destination, args.experiment)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.dry_run:
        print(render_dry_run(plan))
    else:
        print(render_copy_report(plan, copy_plan(plan)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
