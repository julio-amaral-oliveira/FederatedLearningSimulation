import tempfile
import unittest
from pathlib import Path

from experiments.shared.migrate_outputs import build_plan, copy_plan, render_dry_run


class TestMigrateOutputs(unittest.TestCase):
    def test_dry_run_maps_every_file_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            (source / "seed_42").mkdir(parents=True)
            (source / "seed_42" / "agent.json").write_text("agent", encoding="utf-8")
            (source / "seed_42" / "baseline.json").write_text("baseline", encoding="utf-8")
            (source / "summary.json").write_text("summary", encoding="utf-8")

            plan = build_plan(source, destination, "e07-drift-agent")
            rendered = render_dry_run(plan)

            self.assertEqual(plan.file_count, 3)
            self.assertFalse(destination.exists())
            self.assertIn(
                f"would_copy: {source / 'seed_42' / 'agent.json'} -> "
                f"{destination / 'seed_42' / 'agent.json'}",
                rendered,
            )
            self.assertIn("DRY RUN: no files will be copied.", rendered)

    def test_missing_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                build_plan(root / "missing", root / "destination", "e07-drift-agent")

    def test_copy_validates_digests_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            source_file = source / "summary.json"
            source_file.write_text("summary", encoding="utf-8")

            report = copy_plan(build_plan(source, destination, "e07-drift-agent"))

            self.assertEqual(report.file_count, 1)
            self.assertEqual(report.pair_count, 0)
            self.assertEqual(
                (destination / "summary.json").read_text(encoding="utf-8"),
                "summary",
            )
            self.assertEqual(source_file.read_text(encoding="utf-8"), "summary")

    def test_copy_refuses_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "summary.json").write_text("summary", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                copy_plan(build_plan(source, destination, "e07-drift-agent"))

    def test_destination_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                build_plan(source, source / "destination", "e07-drift-agent")
