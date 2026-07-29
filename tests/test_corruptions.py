import unittest

import torch

from src.utils.corruptions import apply_corruption


CORRUPTIONS = ("gaussian_noise", "frosted_glass_blur", "motion_blur", "fog")
SEVERITIES = (1, 2, 3, 4, 5)


def deterministic_image(batch_size: int | None = None) -> torch.Tensor:
    """Return a spatial fixture whose corruption is observable away from clamp edges."""
    horizontal = torch.linspace(0.1, 0.9, 16)
    vertical = torch.linspace(0.1, 0.9, 16).unsqueeze(1)
    base = (horizontal + vertical) / 2
    image = torch.stack((base, base.flip(0), base.flip(1)))
    return image if batch_size is None else image.unsqueeze(0).repeat(batch_size, 1, 1, 1)


class ApplyCorruptionContractTests(unittest.TestCase):
    def test_every_corruption_preserves_tensor_contract_without_mutating_3d_input(self):
        source = deterministic_image()
        original = source.clone()

        for kind in CORRUPTIONS:
            for severity in SEVERITIES:
                with self.subTest(kind=kind, severity=severity):
                    result = apply_corruption(source, kind, severity, seed=17)

                    self.assertEqual(result.shape, source.shape)
                    self.assertEqual(result.dtype, source.dtype)
                    self.assertEqual(result.device, source.device)
                    self.assertTrue(torch.equal(source, original))
                    self.assertGreaterEqual(result.min().item(), 0.0)
                    self.assertLessEqual(result.max().item(), 1.0)
                    self.assertFalse(torch.equal(result, source))

    def test_every_corruption_supports_batched_4d_input(self):
        source = deterministic_image(batch_size=2)

        for kind in CORRUPTIONS:
            with self.subTest(kind=kind):
                result = apply_corruption(source, kind, severity=3, seed=23)

                self.assertEqual(result.shape, source.shape)
                self.assertEqual(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertFalse(torch.equal(result, source))

    def test_seed_makes_every_corruption_reproducible_and_seed_changes_vary_result(self):
        source = deterministic_image()

        for kind in CORRUPTIONS:
            with self.subTest(kind=kind):
                first = apply_corruption(source, kind, severity=3, seed=17)
                repeated = apply_corruption(source, kind, severity=3, seed=17)
                variations = [
                    apply_corruption(source, kind, severity=3, seed=seed)
                    for seed in range(18, 32)
                ]

                self.assertTrue(torch.equal(first, repeated))
                self.assertTrue(any(not torch.equal(first, varied) for varied in variations))

    def test_severity_must_be_an_integer_from_one_through_five(self):
        source = deterministic_image()

        for severity in (0, 6, 1.0, True):
            with self.subTest(severity=severity):
                with self.assertRaises(ValueError):
                    apply_corruption(source, "gaussian_noise", severity, seed=17)

    def test_unknown_corruption_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_corruption(deterministic_image(), "unknown", severity=1, seed=17)

    def test_severity_never_decreases_mean_absolute_change_on_deterministic_fixture(self):
        source = deterministic_image()

        for kind in CORRUPTIONS:
            with self.subTest(kind=kind):
                distances = [
                    (apply_corruption(source, kind, severity, seed=17) - source)
                    .abs()
                    .mean()
                    .item()
                    for severity in SEVERITIES
                ]

                for lower, higher in zip(distances, distances[1:]):
                    self.assertGreaterEqual(higher + 1e-6, lower, distances)


if __name__ == "__main__":
    unittest.main()
