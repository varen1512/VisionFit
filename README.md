# Focus Guard — Eye/Head Position Detector

A computer-vision desktop app that watches your webcam and plays an audio
alert when it detects you've been looking down (e.g., at your phone) for
too long while you're supposed to be working.

## Tech stack

| Library     | What it's used for |
|-------------|---------------------|
| **OpenCV** (`cv2`) | Captures webcam frames, draws the on-screen overlay (status text, landmark dots, progress bar), shows the live window |
| **MediaPipe** | Google's ML framework — runs the `FaceLandmarker` model, which returns 478 3D facial landmark points per frame in real time |
| **NumPy** | Smoothing/averaging the head-tilt score over recent frames |
| **winsound / afplay / paplay** (stdlib + OS tools) | Plays the audio alert — different backend per OS, no extra audio library required |
| **threading** | Runs the alert sound in the background so the video feed never freezes |

## Setup

```bash
pip install -r requirements.txt
python focus_guard.py
```

The first run automatically downloads MediaPipe's face landmark model
(~3–4 MB) into the project folder. After that it works offline.

## Controls

| Key | Action |
|-----|--------|
| `c` | Calibrate — look normally at your screen for 3 seconds to set your personal baseline |
| `+` | Increase sensitivity (alerts on smaller head-down tilts) |
| `-` | Decrease sensitivity |
| `q` | Quit |

**Run calibration first** — everyone's face shape and camera angle is
different, so this tunes the detector to you.

## How the detection works

1. **Face landmarks**: MediaPipe's `FaceLandmarker` returns 478 (x, y, z)
   points describing the face. We only need 3 of them:
   - landmark **1** → nose tip
   - landmark **10** → forehead
   - landmark **152** → chin

2. **Pitch score**: We compute where the nose sits, vertically, between
   the forehead and chin:

   ```
   pitch_score = (nose.y - forehead.y) / (chin.y - forehead.y)
   ```

   - Looking straight at the screen → nose sits roughly in the middle
     (score ≈ 0.45–0.55).
   - Tilting your head down (looking at a phone) → in the 2D camera
     image, the nose appears to move *up* toward the forehead due to
     foreshortening → **score drops**.

3. **Smoothing**: The last 10 frames' scores are averaged so a quick
   blink or glance doesn't trigger a false alert.

4. **Sustained check**: Only after the smoothed score stays below
   `baseline - sensitivity_offset` for **1.5 continuous seconds** does
   an alert fire — with a **6-second cooldown** so it doesn't spam you.

5. **Audio alert**: A 3-tone descending beep, played on a background
   thread so the camera feed doesn't lag.

## Resume bullet point ideas

- *Built "Focus Guard," a real-time computer vision application in
  Python using OpenCV and MediaPipe's FaceLandmarker model to detect
  sustained downward head-tilt (phone-distraction behavior) from
  478-point facial landmarks, with adaptive per-user calibration and
  audio feedback.*
- *Designed a head-pose estimation algorithm using facial landmark
  geometry and rolling-average smoothing to reliably distinguish brief
  glances from sustained distraction, reducing false positives.*

## Possible extensions (good for "what's next" in interviews)

- Log distraction events to a CSV/SQLite file and visualize a daily
  "focus score" with matplotlib.
- Add a Pomodoro-style session timer with statistics.
- Use `EAR` (eye aspect ratio) landmarks to also detect drowsiness.
- Package as a system tray app using `pystray`.
