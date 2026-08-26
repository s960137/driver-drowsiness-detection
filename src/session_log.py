"""Privacy-conscious CSV logging for driver-monitoring validation."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDS = (
    "timestamp",
    "elapsed_seconds",
    "face_detected",
    "calibrated",
    "ear",
    "mar",
    "ear_threshold",
    "mar_threshold",
    "perclos",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "head_away",
    "cumulative_away_seconds",
    "total_blinks",
    "last_bpm",
    "status",
)


class SessionLogger:
    def __init__(self, directory: Path, started_at: float, interval_seconds: float = 0.2) -> None:
        if interval_seconds <= 0:
            raise ValueError("log interval must be positive")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"driver-state-{datetime.now():%Y%m%d-%H%M%S-%f}.csv"
        self.started_at = started_at
        self.interval_seconds = interval_seconds
        self.next_log_at = started_at
        self._file = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, now: float, values: dict[str, Any]) -> bool:
        if now < self.next_log_at:
            return False
        row = {field: values.get(field, "") for field in FIELDS}
        row["timestamp"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        row["elapsed_seconds"] = f"{now - self.started_at:.3f}"
        self._writer.writerow(row)
        self._file.flush()
        self.next_log_at = now + self.interval_seconds
        return True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
