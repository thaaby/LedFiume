# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

LedFiume is a real-time marble tracking system. A webcam detects colored marbles (rosso/blu/giallo), tracks their positions using Kalman + Hungarian algorithm, and maps them onto a 32×32 LED matrix made of 4 WS2812B panels — communicating the frame over serial to an Arduino.

## Running the project

```bash
# Activate the venv (Python 3.14, has all deps including mediapipe, ultralytics, torch)
source venv/bin/activate

# Run the tracker
python TrackPalline.py
```

The `.venv` (Python 3.12) has only the basic deps (opencv, numpy, pygame, pyserial). The `venv` (Python 3.14) is the full environment.

Press `q` in the OpenCV window to quit.

## Install dependencies

```bash
pip install -r requirements.txt
# Full env also needs: mediapipe ultralytics torch scipy sounddevice
```

## Architecture

### Python side — `TrackPalline.py`

Single-file application with these logical sections:

1. **Config block** — Arduino serial parameters, panel layout, serpentine wiring direction
2. **`precompute_led_mapping()`** — Pre-builds a NumPy index array at startup that maps the 32×32 logical matrix to physical LED strip order (accounts for serpentine rows and per-panel orientation)
3. **`TrackedObject`** — Per-marble Kalman filter (4-state: x, y, dx, dy)
4. **`MarbleTracker`** — Manages all live objects; uses `scipy.optimize.linear_sum_assignment` (Hungarian) for ID assignment across frames
5. **`send_matrix_state()`** — Async serial send: skips frame if `arduino_ready=False` so OpenCV/Kalman runs at full speed while Arduino is busy with `FastLED.show()` (~31ms)
6. **`main()`** — Capture loop: resize→blur→HSV segmentation→morphology→contours→tracker update→matrix render→send

### Arduino side — `arduino_palette_sketch/arduino_palette_sketch.ino`

- 500000 baud serial, pin 6, WS2812B GRB, brightness 40
- Protocol: wait for magic header `0xFF 0x4C 0x45`, then read 3072 bytes (1024 LEDs × 3 RGB), call `FastLED.show()`, reply `'K'`
- No background animations, no unnecessary interrupt blocking — loop as fast as possible to avoid dropping serial bytes

### Serial protocol

```
Python → Arduino:  [0xFF][0x4C][0x45] + 3072 bytes (gamma-corrected RGB, physical LED order)
Arduino → Python:  'K'  (frame shown OK)
```

## Physical LED layout

- 4 panels of 8 columns × 32 rows = 32×32 total
- `ARDUINO_PANEL_ORDER = [3, 2, 1, 0]` — panels are wired right-to-left
- `ARDUINO_SERPENTINE_X = True` — odd rows reverse direction
- Gamma correction: `GAMMA = 2.5` applied on the Python side before sending

## Key tuning parameters in `TrackPalline.py`

| Constant | Purpose |
|---|---|
| `SCALE_FACTOR` | Processing resolution (0.5 = half-size, ~4× faster) |
| `COLOR_RANGES` | HSV bounds per color — adjust if marbles aren't detected |
| `max_distance=400` | Max pixel jump between frames before tracker drops a marble |
| `max_disappeared=10` | Frames a marble can go unseen before deregistering |
| `obj.hits < 3` | Temporal filter: ignore detections under 3 consecutive frames |

## Model files present (unused in current main file)

- `yolov8n-seg.pt` — YOLOv8 nano segmentation
- `pose_landmarker_heavy.task`, `pose_landmarker_lite.task` — MediaPipe pose
- `selfie_segmenter.tflite` — MediaPipe selfie segmentation
