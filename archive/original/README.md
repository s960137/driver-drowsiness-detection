# Original 2024 script

`eyes_mouths.py` is the original eye-closure and yawn-alert prototype. It is preserved as historical source.

Use `src/drowsiness_monitor.py` for the cleaned implementation. The newer version avoids webcam-FPS division, emits one event per continuous closure/yawn, and limits repeated audio alerts.
