"""Webcam driver-drowsiness demonstration using EAR and MAR."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from drowsiness_state import DrowsinessTracker
from landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio


RIGHT_EYE = slice(36, 42)
LEFT_EYE = slice(42, 48)
MOUTH = slice(48, 68)
DEFAULT_ALERT = Path(__file__).resolve().parents[1] / "assets" / "alert.wav"


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
    parser = argparse.ArgumentParser(description="Demonstrate driver drowsiness and yawn alerts")
    parser.add_argument("--shape-predictor", required=True, type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--alert-sound", type=Path, default=DEFAULT_ALERT)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--alert-cooldown", type=float, default=3.0)
    parser.add_argument("--ear-threshold", type=float, default=0.23)
    parser.add_argument("--minimum-blink-frames", type=int, default=3)
    parser.add_argument("--fatigue-seconds", type=float, default=1.0)
    parser.add_argument("--mar-threshold", type=float, default=0.75)
    parser.add_argument("--yawn-seconds", type=float, default=0.8)
    parser.add_argument("--blink-rate-threshold", type=float, default=30.0)
    return parser


def run(args: argparse.Namespace) -> None:
    import cv2
    import dlib
    import numpy as np

    if not args.shape_predictor.is_file():
        raise FileNotFoundError(f"Landmark model not found: {args.shape_predictor}")
    if not args.no_audio and not args.alert_sound.is_file():
        raise FileNotFoundError(f"Alert sound not found: {args.alert_sound}")
    if args.alert_cooldown < 0 or args.blink_rate_threshold <= 0:
        raise ValueError("alert cooldown must be non-negative and blink-rate threshold must be positive")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.shape_predictor))
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    started_at = time.monotonic()
    tracker = DrowsinessTracker(
        window_started_at=started_at,
        ear_threshold=args.ear_threshold,
        minimum_blink_frames=args.minimum_blink_frames,
        fatigue_seconds=args.fatigue_seconds,
        mar_threshold=args.mar_threshold,
        yawn_seconds=args.yawn_seconds,
    )
    alerts = AlertPlayer(args.alert_sound, not args.no_audio, args.alert_cooldown)
    status = "MONITORING"
    status_until = started_at
    last_blink_rate = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            now = time.monotonic()
            ear = None
            mar = None

            if faces:
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                shape = predictor(gray, face)
                points = np.array([(shape.part(index).x, shape.part(index).y) for index in range(68)])
                left_eye = points[LEFT_EYE]
                right_eye = points[RIGHT_EYE]
                mouth = points[MOUTH]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)
                events = tracker.update(ear, mar, now)

                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

                if "fatigue" in events:
                    status = "LONG EYE CLOSURE"
                    status_until = now + 2.0
                    alerts.play(now)
                elif "yawn" in events:
                    status = "YAWN DETECTED"
                    status_until = now + 2.0
                    alerts.play(now)
            else:
                tracker.reset_partial()

            rate = tracker.roll_blink_rate(now)
            if rate is not None:
                last_blink_rate = rate
                if rate > args.blink_rate_threshold:
                    status = "HIGH BLINK RATE"
                    status_until = now + 2.0
                    alerts.play(now)

            if now >= status_until:
                status = "MONITORING"

            cv2.putText(frame, f"Blinks: {tracker.total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ear_text = f"EAR: {ear:.2f}" if ear is not None else "EAR: no face"
            mar_text = f"MAR: {mar:.2f}" if mar is not None else "MAR: no face"
            cv2.putText(frame, ear_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, mar_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Last BPM: {last_blink_rate:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            color = (0, 0, 255) if status != "MONITORING" else (0, 150, 0)
            cv2.putText(frame, status, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        alerts.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
