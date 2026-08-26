import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal_metrics import DistractionTracker, PersonalCalibrator, RollingPerclos


class PersonalCalibratorTests(unittest.TestCase):
    def test_builds_personal_thresholds_from_neutral_samples(self) -> None:
        calibrator = PersonalCalibrator(0.0, duration=1.0, minimum_samples=3)
        for values in ((0.30, 0.40, 2.0, -1.0), (0.32, 0.42, 4.0, 1.0), (0.31, 0.41, 3.0, 0.0)):
            calibrator.add(*values)
        self.assertTrue(calibrator.ready(1.0))
        result = calibrator.finish()
        self.assertAlmostEqual(result.ear_threshold, 0.31 * 0.8)
        self.assertAlmostEqual(result.mar_threshold, 0.41 + 0.25)
        self.assertAlmostEqual(result.yaw_offset, 3.0)

    def test_calibration_timer_starts_with_first_valid_face(self) -> None:
        calibrator = PersonalCalibrator(None, duration=2.0, minimum_samples=1)
        self.assertEqual(calibrator.progress(10.0), 0.0)
        calibrator.add(0.3, 0.4, 0.0, 0.0, now=10.0)
        self.assertAlmostEqual(calibrator.progress(11.0), 0.5)

    def test_circular_head_offset_handles_angle_wrap(self) -> None:
        calibrator = PersonalCalibrator(0.0, duration=1.0, minimum_samples=2)
        calibrator.add(0.3, 0.4, 179.0, 179.0)
        calibrator.add(0.3, 0.4, -179.0, -179.0)
        result = calibrator.finish()
        self.assertAlmostEqual(abs(result.yaw_offset), 180.0)


class RollingPerclosTests(unittest.TestCase):
    def test_uses_valid_time_instead_of_frame_count(self) -> None:
        tracker = RollingPerclos(ear_threshold=0.23, window_seconds=10.0, minimum_observation=4.0)
        self.assertIsNone(tracker.update(0.20, 0.0))
        self.assertIsNone(tracker.update(0.20, 2.0))
        self.assertAlmostEqual(tracker.update(0.30, 4.0), 1.0)
        self.assertAlmostEqual(tracker.update(0.30, 8.0), 0.5)
        self.assertAlmostEqual(tracker.mark_missing(10.0), 0.4)

    def test_trims_old_intervals(self) -> None:
        tracker = RollingPerclos(ear_threshold=0.23, window_seconds=10.0, minimum_observation=2.0)
        tracker.update(0.20, 0.0)
        tracker.update(0.30, 4.0)
        tracker.update(0.30, 10.0)
        self.assertAlmostEqual(tracker.mark_missing(12.0), 0.2)


class DistractionTrackerTests(unittest.TestCase):
    def test_reports_continuous_distraction_once(self) -> None:
        tracker = DistractionTracker(long_seconds=3.0, cumulative_seconds=10.0)
        tracker.update(True, 0.0)
        self.assertIn("long_distraction", tracker.update(True, 3.1))
        self.assertNotIn("long_distraction", tracker.update(True, 3.2))

    def test_reports_cumulative_distraction(self) -> None:
        tracker = DistractionTracker(long_seconds=5.0, cumulative_seconds=4.0, window_seconds=30.0)
        tracker.update(True, 0.0)
        tracker.update(False, 2.0)
        tracker.update(True, 3.0)
        self.assertIn("cumulative_distraction", tracker.update(False, 5.0))
        self.assertAlmostEqual(tracker.cumulative_away(5.0), 4.0)

    def test_sustained_attentive_gaze_resets_cumulative_window(self) -> None:
        tracker = DistractionTracker(
            long_seconds=5.0,
            cumulative_seconds=2.0,
            window_seconds=30.0,
            reset_seconds=2.0,
        )
        tracker.update(True, 0.0)
        self.assertIn("cumulative_distraction", tracker.update(False, 2.0))
        self.assertNotIn("cumulative_distraction", tracker.update(False, 4.1))
        self.assertEqual(tracker.cumulative_away(4.1), 0.0)


if __name__ == "__main__":
    unittest.main()
