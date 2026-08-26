import argparse
import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness_monitor import DEFAULT_SCREENSHOT_DIR, build_parser


class ArgumentParserTests(unittest.TestCase):
    def parse(self, *values: str) -> argparse.Namespace:
        return build_parser().parse_args(["--shape-predictor", "model.dat", *values])

    def test_default_face_loss_grace_period(self) -> None:
        self.assertEqual(self.parse().face_loss_grace_seconds, 0.25)

    def test_default_screenshot_directory(self) -> None:
        self.assertEqual(self.parse().screenshot_dir, DEFAULT_SCREENSHOT_DIR)

    def test_accepts_disabled_face_loss_grace_period(self) -> None:
        self.assertEqual(
            self.parse("--face-loss-grace-seconds", "0").face_loss_grace_seconds,
            0.0,
        )

    def test_rejects_non_positive_duration(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--fatigue-seconds", "0")

    def test_rejects_zero_minimum_blink_frames(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--minimum-blink-frames", "0")

    def test_rejects_non_finite_float(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--face-loss-grace-seconds", "nan")


if __name__ == "__main__":
    unittest.main()
