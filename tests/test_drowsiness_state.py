import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness_state import DrowsinessTracker


class DrowsinessTrackerTests(unittest.TestCase):
    def new_tracker(self) -> DrowsinessTracker:
        return DrowsinessTracker(
            window_started_at=0.0,
            ear_threshold=0.23,
            minimum_blink_frames=3,
            fatigue_seconds=1.0,
            mar_threshold=0.75,
            yawn_seconds=0.8,
        )

    def test_counts_blink_on_reopening(self) -> None:
        tracker = self.new_tracker()
        tracker.update(0.20, 0.30, 0.0)
        tracker.update(0.20, 0.30, 0.1)
        tracker.update(0.20, 0.30, 0.2)
        events = tracker.update(0.30, 0.30, 0.3)
        self.assertIn("blink", events)
        self.assertEqual(tracker.total_blinks, 1)

    def test_long_closure_reports_fatigue_once(self) -> None:
        tracker = self.new_tracker()
        tracker.update(0.20, 0.30, 0.0)
        self.assertIn("fatigue", tracker.update(0.20, 0.30, 1.1))
        self.assertNotIn("fatigue", tracker.update(0.20, 0.30, 1.2))

    def test_yawn_reports_once_until_mouth_closes(self) -> None:
        tracker = self.new_tracker()
        tracker.update(0.30, 0.80, 0.0)
        self.assertIn("yawn", tracker.update(0.30, 0.80, 0.9))
        self.assertNotIn("yawn", tracker.update(0.30, 0.80, 1.0))
        tracker.update(0.30, 0.30, 1.1)
        tracker.update(0.30, 0.80, 1.2)
        self.assertIn("yawn", tracker.update(0.30, 0.80, 2.1))

    def test_blink_rate_rolls_after_one_minute(self) -> None:
        tracker = self.new_tracker()
        tracker.window_blinks = 6
        self.assertAlmostEqual(tracker.roll_blink_rate(60.0), 6.0)
        self.assertEqual(tracker.window_blinks, 0)

    def test_tracking_loss_discards_partial_blink(self) -> None:
        tracker = self.new_tracker()
        for now in (0.0, 0.1, 0.2):
            tracker.update(0.20, 0.30, now)
        tracker.reset_partial()
        self.assertNotIn("blink", tracker.update(0.30, 0.30, 0.3))


if __name__ == "__main__":
    unittest.main()
