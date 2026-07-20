import unittest

import numpy as np

from src.utils.drift_detector import ADWIN


class TestADWIN(unittest.TestCase):
    def test_stays_quiet_on_stationary_standard_normal(self):
        """Given a stream of N(0,1) samples, ADWIN must not raise drift."""
        # Given: a fresh ADWIN detector with delta=0.002
        detector = ADWIN(delta=0.002)
        rng = np.random.default_rng(42)
        stationary = rng.normal(loc=0.0, scale=1.0, size=200)

        # When: feeding 200 stationary samples
        for value in stationary:
            drift_detected, _ = detector.update(float(value))

            # Then: no drift is reported at any point
            self.assertFalse(drift_detected)

    def test_detects_step_change_to_shifted_normal(self):
        """Given stationary N(0,1) then N(1,1), ADWIN must report drift."""
        # Given: a fresh ADWIN detector and a stationary prefix
        detector = ADWIN(delta=0.002)
        rng = np.random.default_rng(42)
        prefix = rng.normal(loc=0.0, scale=1.0, size=200)

        for value in prefix:
            detector.update(float(value))

        # When: the distribution shifts to N(1,1)
        detected = False
        for _ in range(500):
            value = rng.normal(loc=1.0, scale=1.0)
            drift_detected, _ = detector.update(float(value))
            if drift_detected:
                detected = True
                break

        # Then: drift is reported after the shift
        self.assertTrue(detected)


if __name__ == "__main__":
    unittest.main()
