"""Estimate coarse driver head orientation from Dlib facial landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


LANDMARK_INDEXES = np.array([30, 8, 36, 45, 48, 54])
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float


def angular_difference(value: float, reference: float) -> float:
    """Return the shortest signed angular difference in degrees."""
    return (value - reference + 180.0) % 360.0 - 180.0


def estimate_head_pose(landmarks: np.ndarray, frame_width: int, frame_height: int) -> HeadPose | None:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("facial landmarks must have shape (68, 2) and finite values")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    focal_length = float(frame_width)
    camera_matrix = np.array(
        [
            (focal_length, 0.0, frame_width / 2.0),
            (0.0, focal_length, frame_height / 2.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    image_points = points[LANDMARK_INDEXES]
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return HeadPose(pitch=float(angles[0]), yaw=float(angles[1]), roll=float(angles[2]))
