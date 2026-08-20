"""Geometry helpers for facial landmark measurements."""

from __future__ import annotations

import numpy as np


def _points(value: np.ndarray, expected: int, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.shape != (expected, 2):
        raise ValueError(f"{label} must have shape ({expected}, 2), got {points.shape}")
    return points


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Return the Eye Aspect Ratio for six ordered eye landmarks."""
    points = _points(eye, 6, "eye")
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal == 0:
        raise ValueError("eye horizontal distance must be greater than zero")
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Return the Mouth Aspect Ratio for Dlib landmarks 48 through 67."""
    points = _points(mouth, 20, "mouth")
    vertical_1 = np.linalg.norm(points[13] - points[19])
    vertical_2 = np.linalg.norm(points[14] - points[18])
    vertical_3 = np.linalg.norm(points[15] - points[17])
    horizontal = np.linalg.norm(points[12] - points[16])
    if horizontal == 0:
        raise ValueError("mouth horizontal distance must be greater than zero")
    return float((vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal))
