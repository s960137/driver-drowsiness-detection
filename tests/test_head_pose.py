import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from head_pose import LANDMARK_INDEXES, MODEL_POINTS, angular_difference, estimate_head_pose


class HeadPoseTests(unittest.TestCase):
    def test_estimates_neutral_synthetic_pose(self) -> None:
        width, height = 640, 480
        camera_matrix = np.array(((width, 0, width / 2), (0, width, height / 2), (0, 0, 1)), dtype=float)
        projected, _ = cv2.projectPoints(
            MODEL_POINTS,
            np.zeros((3, 1)),
            np.array(((0.0,), (0.0,), (1000.0,))),
            camera_matrix,
            np.zeros((4, 1)),
        )
        landmarks = np.zeros((68, 2), dtype=float)
        landmarks[LANDMARK_INDEXES] = projected.reshape(-1, 2)
        pose = estimate_head_pose(landmarks, width, height)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.pitch, 0.0, delta=0.1)
        self.assertAlmostEqual(pose.yaw, 0.0, delta=0.1)

    def test_rejects_invalid_landmarks(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            estimate_head_pose(np.zeros((6, 2)), 640, 480)

    def test_angular_difference_wraps_at_180_degrees(self) -> None:
        self.assertAlmostEqual(angular_difference(179.0, -179.0), -2.0)
        self.assertAlmostEqual(angular_difference(-179.0, 179.0), 2.0)


if __name__ == "__main__":
    unittest.main()
