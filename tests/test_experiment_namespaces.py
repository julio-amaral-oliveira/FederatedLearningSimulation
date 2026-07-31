import importlib
import unittest


class TestExperimentNamespaces(unittest.TestCase):
    def test_all_planned_namespaces_are_importable(self):
        modules = [
            "experiments.e01_static",
            "experiments.e02_ablation.run",
            "experiments.e02_ablation.plot",
            "experiments.e03_timeout",
            "experiments.e04_static_comparison.compare",
            "experiments.e04_static_comparison.ui",
            "experiments.e05_temporal_drift.run",
            "experiments.e05_temporal_drift.plot",
            "experiments.e05_temporal_drift.ui",
            "experiments.e06_drift_agent_prototype.prototype",
            "experiments.e06_drift_agent_prototype.ui",
            "experiments.e07_drift_agent.episode",
            "experiments.e07_drift_agent.calibrate",
            "experiments.e07_drift_agent.run_matrix",
            "experiments.e07_drift_agent.plot",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                if hasattr(module, "main"):
                    self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
