"""Calibrated webcam demonstration for driver fatigue and distraction."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from math import isfinite
from pathlib import Path

from drowsiness_state import DrowsinessTracker
from head_pose import angular_difference, estimate_head_pose
from landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio
from session_log import SessionLogger
from temporal_metrics import DistractionTracker, PersonalCalibrator, RollingPerclos


RIGHT_EYE = slice(36, 42)
LEFT_EYE = slice(42, 48)
MOUTH = slice(48, 68)
DEFAULT_ALERT = Path(__file__).resolve().parents[1] / "assets" / "alert.wav"
DEFAULT_SCREENSHOT_DIR = Path(__file__).resolve().parents[1] / "captures"
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def positive_float(value: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite number that is zero or greater")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def unit_interval(value: str) -> float:
    number = float(value)
    if not isfinite(number) or not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be a finite number between zero and one")
    return number


class AlertPlayer:
    def __init__(self, sound_path: Path, enabled: bool, cooldown: float) -> None:
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_played_at = float("-inf")
        self.pygame = None
        self.sound = None

        if not enabled:
            return
        try:
            import pygame

            pygame.mixer.init()
            self.pygame = pygame
            self.sound = pygame.mixer.Sound(str(sound_path))
        except Exception as error:
            self.enabled = False
            print(f"[WARN] Audio alert disabled: {error}")

    def play(self, now: float) -> None:
        if not self.enabled or self.sound is None:
            return
        if now - self.last_played_at < self.cooldown:
            return
        self.sound.play()
        self.last_played_at = now

    def close(self) -> None:
        if self.pygame is not None:
            self.pygame.mixer.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate calibrated driver fatigue and distraction alerts"
    )
    parser.add_argument("--shape-predictor", required=True, type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--alert-sound", type=Path, default=DEFAULT_ALERT)
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log-interval", type=positive_float, default=0.2)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--alert-cooldown", type=non_negative_float, default=3.0)
    parser.add_argument("--ear-threshold", type=positive_float, default=0.23)
    parser.add_argument("--minimum-blink-frames", type=positive_int, default=3)
    parser.add_argument("--fatigue-seconds", type=positive_float, default=1.0)
    parser.add_argument("--sleep-seconds", type=positive_float, default=3.0)
    parser.add_argument("--unresponsive-seconds", type=positive_float, default=6.0)
    parser.add_argument("--mar-threshold", type=positive_float, default=0.75)
    parser.add_argument("--yawn-seconds", type=positive_float, default=0.8)
    parser.add_argument("--blink-rate-threshold", type=positive_float, default=30.0)
    parser.add_argument("--perclos-window", type=positive_float, default=60.0)
    parser.add_argument("--perclos-min-observation", type=positive_float, default=20.0)
    parser.add_argument("--perclos-threshold", type=unit_interval, default=0.40)
    parser.add_argument("--calibration-seconds", type=non_negative_float, default=10.0)
    parser.add_argument("--calibration-min-samples", type=positive_int, default=30)
    parser.add_argument("--ear-baseline-ratio", type=unit_interval, default=0.80)
    parser.add_argument("--mar-baseline-margin", type=positive_float, default=0.25)
    parser.add_argument("--head-yaw-threshold", type=positive_float, default=25.0)
    parser.add_argument("--head-pitch-threshold", type=positive_float, default=20.0)
    parser.add_argument("--long-distraction-seconds", type=positive_float, default=3.0)
    parser.add_argument("--cumulative-distraction-seconds", type=positive_float, default=10.0)
    parser.add_argument("--distraction-window", type=positive_float, default=30.0)
    parser.add_argument(
        "--face-loss-grace-seconds",
        type=non_negative_float,
        default=0.25,
        help="retain partial eye/mouth state during brief face-detection gaps",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    import cv2
    import dlib
    import numpy as np

    if not args.shape_predictor.is_file():
        raise FileNotFoundError(f"Landmark model not found: {args.shape_predictor}")
    if not args.no_audio and not args.alert_sound.is_file():
        raise FileNotFoundError(f"Alert sound not found: {args.alert_sound}")
    if not args.fatigue_seconds < args.sleep_seconds < args.unresponsive_seconds:
        raise ValueError("closure durations must increase from microsleep to sleep to unresponsive")
    if args.perclos_min_observation > args.perclos_window:
        raise ValueError("minimum PERCLOS observation cannot exceed its window")
    if args.cumulative_distraction_seconds > args.distraction_window:
        raise ValueError("cumulative distraction threshold cannot exceed its window")

    started_at = time.monotonic()
    tracker = DrowsinessTracker(
        window_started_at=started_at,
        ear_threshold=args.ear_threshold,
        minimum_blink_frames=args.minimum_blink_frames,
        fatigue_seconds=args.fatigue_seconds,
        sleep_seconds=args.sleep_seconds,
        unresponsive_seconds=args.unresponsive_seconds,
        mar_threshold=args.mar_threshold,
        yawn_seconds=args.yawn_seconds,
    )
    perclos_tracker = RollingPerclos(
        ear_threshold=args.ear_threshold,
        window_seconds=args.perclos_window,
        minimum_observation=args.perclos_min_observation,
    )
    distraction_tracker = DistractionTracker(
        long_seconds=args.long_distraction_seconds,
        cumulative_seconds=args.cumulative_distraction_seconds,
        window_seconds=args.distraction_window,
    )
    calibrator = None
    if args.calibration_seconds > 0:
        calibrator = PersonalCalibrator(
            started_at=None,
            duration=args.calibration_seconds,
            minimum_samples=args.calibration_min_samples,
            ear_ratio=args.ear_baseline_ratio,
            mar_margin=args.mar_baseline_margin,
        )
    calibrated = calibrator is None
    pose_calibrated = False
    yaw_offset = 0.0
    pitch_offset = 0.0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.shape_predictor))
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    alerts = AlertPlayer(args.alert_sound, not args.no_audio, args.alert_cooldown)
    try:
        logger = None if args.no_log else SessionLogger(args.log_dir, started_at, args.log_interval)
    except Exception:
        camera.release()
        alerts.close()
        raise
    if logger is not None:
        print(f"[INFO] Session log: {logger.path.resolve()}")

    status = "MONITORING" if calibrated else "CALIBRATING"
    status_until = started_at
    last_blink_rate = 0.0
    face_missing_since = None
    perclos_value = None
    perclos_reported = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[WARN] Camera frame could not be read; stopping monitor.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            now = time.monotonic()
            ear = None
            mar = None
            pitch = None
            yaw = None
            roll = None
            head_away = False
            events: set[str] = set()

            if faces:
                face_missing_since = None
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                shape = predictor(gray, face)
                points = np.array(
                    [(shape.part(index).x, shape.part(index).y) for index in range(68)],
                    dtype=np.int32,
                )
                left_eye = points[LEFT_EYE]
                right_eye = points[RIGHT_EYE]
                mouth = points[MOUTH]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)
                pose = estimate_head_pose(points, frame.shape[1], frame.shape[0])
                if pose is not None:
                    pitch, yaw, roll = pose.pitch, pose.yaw, pose.roll

                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

                if not calibrated and calibrator is not None:
                    tracker.reset_partial()
                    perclos_tracker.mark_missing(now)
                    distraction_tracker.mark_missing(now)
                    if pitch is not None and yaw is not None:
                        calibrator.add(ear, mar, yaw, pitch, now)
                    if calibrator.ready(now):
                        result = calibrator.finish()
                        tracker.set_thresholds(result.ear_threshold, result.mar_threshold)
                        perclos_tracker.set_ear_threshold(result.ear_threshold)
                        tracker.window_started_at = now
                        tracker.window_blinks = 0
                        yaw_offset = result.yaw_offset
                        pitch_offset = result.pitch_offset
                        calibrated = True
                        pose_calibrated = True
                        calibrator = None
                        status = "CALIBRATION COMPLETE"
                        status_until = now + 2.0
                        print(
                            "[INFO] Calibration complete: "
                            f"EAR<{result.ear_threshold:.3f}, MAR>{result.mar_threshold:.3f}, "
                            f"samples={result.sample_count}"
                        )
                else:
                    events |= tracker.update(ear, mar, now)
                    perclos_value = perclos_tracker.update(ear, now)
                    if pitch is not None and yaw is not None and pose_calibrated:
                        relative_pitch = angular_difference(pitch, pitch_offset)
                        relative_yaw = angular_difference(yaw, yaw_offset)
                        head_away = (
                            abs(relative_yaw) >= args.head_yaw_threshold
                            or abs(relative_pitch) >= args.head_pitch_threshold
                        )
                        events |= distraction_tracker.update(head_away, now)
                    else:
                        distraction_tracker.mark_missing(now)
            else:
                perclos_value = perclos_tracker.mark_missing(now)
                distraction_tracker.mark_missing(now)
                if face_missing_since is None:
                    face_missing_since = now
                if now - face_missing_since >= args.face_loss_grace_seconds:
                    tracker.reset_partial()

            if calibrated and perclos_value is not None:
                if perclos_value >= args.perclos_threshold and not perclos_reported:
                    events.add("drowsy")
                    perclos_reported = True
                elif perclos_value < args.perclos_threshold * 0.8:
                    perclos_reported = False

            rate = tracker.roll_blink_rate(now)
            if rate is not None:
                last_blink_rate = rate
                if rate > args.blink_rate_threshold:
                    events.add("high_blink_rate")

            event_statuses = (
                ("unresponsive", "UNRESPONSIVE"),
                ("sleep", "SLEEP"),
                ("microsleep", "MICROSLEEP"),
                ("drowsy", "DROWSY"),
                ("long_distraction", "DISTRACTED"),
                ("cumulative_distraction", "DISTRACTED"),
                ("yawn", "YAWN DETECTED"),
                ("high_blink_rate", "HIGH BLINK RATE"),
            )
            for event_name, event_status in event_statuses:
                if event_name in events:
                    status = event_status
                    status_until = now + (5.0 if event_name == "unresponsive" else 2.0)
                    alerts.play(now)
                    break

            if not calibrated and calibrator is not None:
                progress = int(calibrator.progress(now) * 100)
                if not faces:
                    status = "CALIBRATION: NO FACE"
                elif progress >= 100:
                    status = "CALIBRATION: HOLD STILL"
                else:
                    status = f"CALIBRATING {progress}%"
            elif now >= status_until:
                status = "MONITORING" if faces else "NO FACE"

            cv2.putText(frame, f"Blinks: {tracker.total_blinks}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ear_text = f"EAR: {ear:.2f}" if ear is not None else "EAR: no face"
            mar_text = f"MAR: {mar:.2f}" if mar is not None else "MAR: no face"
            perclos_text = f"PERCLOS: {perclos_value:.0%}" if perclos_value is not None else "PERCLOS: collecting"
            pose_text = (
                f"Head P/Y: {angular_difference(pitch, pitch_offset):+.0f}/"
                f"{angular_difference(yaw, yaw_offset):+.0f}"
                if pitch is not None and yaw is not None and pose_calibrated
                else ("Head P/Y: uncalibrated" if pitch is not None else "Head P/Y: unavailable")
            )
            cv2.putText(frame, ear_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, mar_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, perclos_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            cv2.putText(frame, pose_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            cv2.putText(frame, f"Last BPM: {last_blink_rate:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            if status in {"UNRESPONSIVE", "SLEEP", "MICROSLEEP", "DROWSY", "YAWN DETECTED", "HIGH BLINK RATE"}:
                color = (0, 0, 255)
            elif status in {"DISTRACTED", "NO FACE", "CALIBRATION: NO FACE", "CALIBRATION: HOLD STILL"}:
                color = (0, 165, 255)
            else:
                color = (0, 150, 0)
            cv2.putText(frame, status, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if logger is not None:
                logger.write(
                    now,
                    {
                        "face_detected": bool(faces),
                        "calibrated": calibrated,
                        "ear": "" if ear is None else f"{ear:.5f}",
                        "mar": "" if mar is None else f"{mar:.5f}",
                        "ear_threshold": f"{tracker.ear_threshold:.5f}",
                        "mar_threshold": f"{tracker.mar_threshold:.5f}",
                        "perclos": "" if perclos_value is None else f"{perclos_value:.5f}",
                        "head_pitch": ""
                        if pitch is None or not pose_calibrated
                        else f"{angular_difference(pitch, pitch_offset):.3f}",
                        "head_yaw": ""
                        if yaw is None or not pose_calibrated
                        else f"{angular_difference(yaw, yaw_offset):.3f}",
                        "head_roll": "" if roll is None else f"{roll:.3f}",
                        "head_away": head_away,
                        "cumulative_away_seconds": f"{distraction_tracker.cumulative_away(now):.3f}",
                        "total_blinks": tracker.total_blinks,
                        "last_bpm": f"{last_blink_rate:.3f}",
                        "status": status,
                    },
                )

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                args.screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot = args.screenshot_dir / f"drowsiness-demo-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
                encoded, image = cv2.imencode(".png", frame)
                if encoded:
                    image.tofile(screenshot)
                    print(f"[INFO] Screenshot saved: {screenshot.resolve()}")
                else:
                    print("[WARN] Screenshot could not be encoded.")
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        alerts.close()
        if logger is not None:
            logger.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
