# Driver Drowsiness Detection

An archived computer-vision prototype that combines eye closure, blink rate, and mouth opening to demonstrate driver-fatigue and yawn alerts. It uses OpenCV, Dlib's 68-point facial landmarks, Eye Aspect Ratio (EAR), and Mouth Aspect Ratio (MAR).

> Research demonstration only. This software has not been validated for real vehicles or safety-critical use. Do not rely on it to prevent a crash.

## Features

- Detects a face and tracks the eye and mouth landmarks.
- Counts one blink per complete close/reopen cycle.
- Raises a fatigue event after the eyes remain closed for a configurable duration.
- Raises a yawn event after the mouth remains open beyond the configured MAR and duration thresholds.
- Calculates blink rate over one-minute windows.
- Plays an alert with a cooldown so the sound is not restarted every frame.

## Repository layout

```text
.
├── archive/original/       # Original 2024 eyes_mouths.py
├── assets/                 # Alert sound from the original prototype
├── src/                    # Cleaned monitor, metrics, and state logic
└── tests/                  # Unit tests that do not require a webcam
```

The original prototype is preserved for project history. The runnable `src` version fixes repeated blink/yawn events and measures long eye closure with a monotonic clock rather than webcam FPS.

## Setup

The cleaned implementation targets Python 3.11.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and decompress Dlib's official landmark model:

<http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>

The model is intentionally excluded because it is about 95 MiB. Dlib states that it was trained on the iBUG 300-W dataset, whose license excludes commercial use. Review the upstream terms before using it outside research or education.

## Run

```bash
python src/drowsiness_monitor.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

Press `q` to stop. Audio can be disabled with `--no-audio`. Thresholds such as `--ear-threshold`, `--fatigue-seconds`, `--mar-threshold`, and `--yawn-seconds` are configurable.

The defaults are demonstration values inherited from the prototype and have not been calibrated for a particular driver, camera, lighting condition, or vehicle environment.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Publication notes

- Do not publish `.venv`, a locally built Dlib source tree, platform-specific wheel files, or the landmark model.
- Confirm that `assets/alert.wav` is owned by the project or licensed for redistribution before making the repository public.
- The included program processes webcam frames locally and does not transmit them.

## Attribution and license

The EAR method follows Soukupová and Čech (2016). The original prototype adapts code from Adrian Rosebrock's PyImageSearch blink-detection tutorial. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No project-wide open-source license has been selected. Unless a license is added by the project owner, the original project code and media remain under the owner's default copyright; third-party components remain under their respective terms.
