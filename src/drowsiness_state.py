"""Time- and frame-based state for blink, fatigue, and yawn events."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class DrowsinessTracker:
    window_started_at: float
    ear_threshold: float = 0.23
    minimum_blink_frames: int = 3
    fatigue_seconds: float = 1.0
    mar_threshold: float = 0.75
    yawn_seconds: float = 0.8
    total_blinks: int = 0
    window_blinks: int = 0
    closed_frames: int = 0
    eyes_closed_at: float | None = None
    mouth_opened_at: float | None = None
    fatigue_reported: bool = False
    yawn_reported: bool = False
    last_updated_at: float | None = None

    def __post_init__(self) -> None:
        if self.ear_threshold <= 0 or self.mar_threshold <= 0:
            raise ValueError("EAR and MAR thresholds must be positive")
        if self.minimum_blink_frames < 1:
            raise ValueError("minimum_blink_frames must be at least 1")
        if self.fatigue_seconds <= 0 or self.yawn_seconds <= 0:
            raise ValueError("event durations must be positive")

    def update(self, ear: float, mar: float, now: float) -> set[str]:
        """Update tracking and return newly triggered event names."""
        if not all(isfinite(value) for value in (ear, mar, now)):
            raise ValueError("EAR, MAR, and timestamp values must be finite")
        if self.last_updated_at is not None and now < self.last_updated_at:
            raise ValueError("timestamps must be monotonic")
        self.last_updated_at = now
        events: set[str] = set()

        if ear < self.ear_threshold:
            if self.eyes_closed_at is None:
                self.eyes_closed_at = now
            self.closed_frames += 1
            if not self.fatigue_reported and now - self.eyes_closed_at >= self.fatigue_seconds:
                events.add("fatigue")
                self.fatigue_reported = True
        else:
            if self.closed_frames >= self.minimum_blink_frames:
                self.total_blinks += 1
                self.window_blinks += 1
                events.add("blink")
            self.closed_frames = 0
            self.eyes_closed_at = None
            self.fatigue_reported = False

        if mar > self.mar_threshold:
            if self.mouth_opened_at is None:
                self.mouth_opened_at = now
            if not self.yawn_reported and now - self.mouth_opened_at >= self.yawn_seconds:
                events.add("yawn")
                self.yawn_reported = True
        else:
            self.mouth_opened_at = None
            self.yawn_reported = False

        return events

    def roll_blink_rate(self, now: float, window_seconds: float = 60.0) -> float | None:
        """Return blinks per minute when a window completes, then start a new window."""
        if not isfinite(now) or not isfinite(window_seconds):
            raise ValueError("timestamp and blink-rate window must be finite")
        if window_seconds <= 0:
            raise ValueError("blink-rate window must be positive")
        if now < self.window_started_at:
            raise ValueError("timestamp cannot precede the blink-rate window")
        elapsed = now - self.window_started_at
        if elapsed < window_seconds:
            return None
        rate = self.window_blinks / (elapsed / 60.0)
        self.window_started_at = now
        self.window_blinks = 0
        return rate

    def reset_partial(self) -> None:
        """Discard unfinished eye and mouth events after face tracking is lost."""
        self.closed_frames = 0
        self.eyes_closed_at = None
        self.mouth_opened_at = None
        self.fatigue_reported = False
        self.yawn_reported = False
