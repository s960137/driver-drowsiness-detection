"""Time-weighted driver-state metrics and personal calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, degrees, isfinite, radians, sin
from statistics import median


@dataclass(frozen=True)
class CalibrationResult:
    ear_threshold: float
    mar_threshold: float
    yaw_offset: float
    pitch_offset: float
    sample_count: int


class PersonalCalibrator:
    """Estimate neutral facial thresholds while the driver looks forward."""

    def __init__(
        self,
        started_at: float | None = None,
        duration: float = 10.0,
        minimum_samples: int = 30,
        ear_ratio: float = 0.80,
        mar_margin: float = 0.25,
    ) -> None:
        if duration <= 0 or minimum_samples < 1:
            raise ValueError("calibration duration and sample count must be positive")
        if not 0 < ear_ratio < 1 or mar_margin <= 0:
            raise ValueError("calibration EAR ratio and MAR margin are invalid")
        self.started_at = started_at
        self.duration = duration
        self.minimum_samples = minimum_samples
        self.ear_ratio = ear_ratio
        self.mar_margin = mar_margin
        self.ear_samples: list[float] = []
        self.mar_samples: list[float] = []
        self.yaw_samples: list[float] = []
        self.pitch_samples: list[float] = []

    def add(self, ear: float, mar: float, yaw: float, pitch: float, now: float | None = None) -> None:
        values = (ear, mar, yaw, pitch) if now is None else (ear, mar, yaw, pitch, now)
        if not all(isfinite(value) for value in values):
            return
        if self.started_at is None and now is not None:
            self.started_at = now
        self.ear_samples.append(ear)
        self.mar_samples.append(mar)
        self.yaw_samples.append(yaw)
        self.pitch_samples.append(pitch)

    def progress(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        return min(max((now - self.started_at) / self.duration, 0.0), 1.0)

    def ready(self, now: float) -> bool:
        return self.progress(now) >= 1.0 and len(self.ear_samples) >= self.minimum_samples

    def finish(self) -> CalibrationResult:
        if len(self.ear_samples) < self.minimum_samples:
            raise ValueError("not enough valid face samples for calibration")
        return CalibrationResult(
            ear_threshold=median(self.ear_samples) * self.ear_ratio,
            mar_threshold=median(self.mar_samples) + self.mar_margin,
            yaw_offset=_circular_mean(self.yaw_samples),
            pitch_offset=_circular_mean(self.pitch_samples),
            sample_count=len(self.ear_samples),
        )


def _circular_mean(values: list[float]) -> float:
    x = sum(cos(radians(value)) for value in values)
    y = sum(sin(radians(value)) for value in values)
    return degrees(atan2(y, x))


class RollingPerclos:
    """Time-weighted fraction of valid observation time with closed eyes."""

    def __init__(self, ear_threshold: float, window_seconds: float = 60.0, minimum_observation: float = 20.0) -> None:
        if ear_threshold <= 0 or window_seconds <= 0 or minimum_observation <= 0:
            raise ValueError("PERCLOS thresholds and durations must be positive")
        if minimum_observation > window_seconds:
            raise ValueError("minimum PERCLOS observation cannot exceed its window")
        self.ear_threshold = ear_threshold
        self.window_seconds = window_seconds
        self.minimum_observation = minimum_observation
        self._intervals: deque[tuple[float, float, bool]] = deque()
        self._last_timestamp: float | None = None
        self._last_closed: bool | None = None

    def set_ear_threshold(self, value: float) -> None:
        if not isfinite(value) or value <= 0:
            raise ValueError("EAR threshold must be positive and finite")
        self.ear_threshold = value

    def update(self, ear: float, now: float) -> float | None:
        if not isfinite(ear):
            raise ValueError("EAR must be finite")
        self._record_until(now)
        self._last_timestamp = now
        self._last_closed = ear < self.ear_threshold
        return self.value(now)

    def mark_missing(self, now: float) -> float | None:
        self._record_until(now)
        self._last_timestamp = None
        self._last_closed = None
        return self.value(now)

    def value(self, now: float) -> float | None:
        self._trim(now)
        observed = sum(end - start for start, end, _ in self._intervals)
        if observed < self.minimum_observation:
            return None
        closed = sum(end - start for start, end, is_closed in self._intervals if is_closed)
        return closed / observed

    def _record_until(self, now: float) -> None:
        if not isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None:
            if now < self._last_timestamp:
                raise ValueError("timestamps must be monotonic")
            if now > self._last_timestamp and self._last_closed is not None:
                self._intervals.append((self._last_timestamp, now, self._last_closed))
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._intervals and self._intervals[0][1] <= cutoff:
            self._intervals.popleft()
        if self._intervals and self._intervals[0][0] < cutoff:
            _, end, closed = self._intervals[0]
            self._intervals[0] = (cutoff, end, closed)


class DistractionTracker:
    """Track continuous and cumulative head-away time."""

    def __init__(
        self,
        long_seconds: float = 3.0,
        cumulative_seconds: float = 10.0,
        window_seconds: float = 30.0,
        reset_seconds: float = 2.0,
    ) -> None:
        if min(long_seconds, cumulative_seconds, window_seconds, reset_seconds) <= 0:
            raise ValueError("distraction durations must be positive")
        self.long_seconds = long_seconds
        self.cumulative_seconds = cumulative_seconds
        self.window_seconds = window_seconds
        self.reset_seconds = reset_seconds
        self._intervals: deque[tuple[float, float, bool]] = deque()
        self._last_timestamp: float | None = None
        self._last_away: bool | None = None
        self._away_started_at: float | None = None
        self._attentive_started_at: float | None = None
        self._long_reported = False
        self._cumulative_reported = False

    def update(self, away: bool, now: float) -> set[str]:
        self._record_until(now)
        events: set[str] = set()

        if away:
            if self._last_away is not True:
                self._away_started_at = now
                self._long_reported = False
            self._attentive_started_at = None
        else:
            if self._last_away is not False:
                self._attentive_started_at = now
            self._away_started_at = None
            self._long_reported = False
            if self._attentive_started_at is not None and now - self._attentive_started_at >= self.reset_seconds:
                self._intervals.clear()
                self._cumulative_reported = False
                self._attentive_started_at = None

        self._last_timestamp = now
        self._last_away = away

        if away and self._away_started_at is not None:
            if not self._long_reported and now - self._away_started_at >= self.long_seconds:
                events.add("long_distraction")
                self._long_reported = True
        if not self._cumulative_reported and self.cumulative_away(now) >= self.cumulative_seconds:
            events.add("cumulative_distraction")
            self._cumulative_reported = True
        return events

    def mark_missing(self, now: float) -> None:
        self._record_until(now)
        self._last_timestamp = None
        self._last_away = None
        self._away_started_at = None
        self._attentive_started_at = None
        self._long_reported = False

    def cumulative_away(self, now: float) -> float:
        self._trim(now)
        return sum(end - start for start, end, away in self._intervals if away)

    def _record_until(self, now: float) -> None:
        if not isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None:
            if now < self._last_timestamp:
                raise ValueError("timestamps must be monotonic")
            if now > self._last_timestamp and self._last_away is not None:
                self._intervals.append((self._last_timestamp, now, self._last_away))
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._intervals and self._intervals[0][1] <= cutoff:
            self._intervals.popleft()
        if self._intervals and self._intervals[0][0] < cutoff:
            _, end, away = self._intervals[0]
            self._intervals[0] = (cutoff, end, away)
