import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio


class LandmarkMetricsTests(unittest.TestCase):
    def test_eye_aspect_ratio(self) -> None:
        eye = np.array([[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]])
        self.assertAlmostEqual(eye_aspect_ratio(eye), 0.5)

    def test_mouth_aspect_ratio(self) -> None:
        mouth = np.zeros((20, 2), dtype=float)
        mouth[12], mouth[16] = (0, 0), (6, 0)
        mouth[13], mouth[19] = (1, 2), (1, -2)
        mouth[14], mouth[18] = (3, 2), (3, -2)
        mouth[15], mouth[17] = (5, 2), (5, -2)
        self.assertAlmostEqual(mouth_aspect_ratio(mouth), 1.0)


if __name__ == "__main__":
    unittest.main()
