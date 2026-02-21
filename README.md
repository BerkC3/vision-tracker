# Vision-Track

Real-time traffic analysis system powered by Ultralytics YOLO and BoT-SORT. Detects vehicles, tracks them across frames, estimates speed, and flags lane violations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Vehicle Detection** - Ultralytics YOLO (YOLOv8 or YOLO11, nano to extra-large) with COCO vehicle classes (car, motorcycle, bus, truck)
- **Multi-Object Tracking** - BoT-SORT with re-identification for robust tracking through occlusions
- **Speed Estimation** - Two-line crossing method with perspective-aware polyline paths
- **Lane Violation Detection** - Multiple named ROI polygon zones (Restricted Zone 1, 2, …); each vehicle-zone pair is recorded exactly once and logged to CSV
- **Interactive Calibration** - Draw speed lines and restricted zones on a configurable stable frame (skips initial camera shake)
- **Dual Interface** - CLI with OpenCV display + Streamlit web dashboard
- **Flexible Input** - Local video files, webcam, or YouTube URLs (via yt-dlp)

## Architecture

```
main.py                 CLI entry point
ui/app.py               Streamlit web UI

core/
  tracker.py            YOLO + BoT-SORT detection & tracking
  speed_estimator.py    Two-line speed measurement
  violation.py          ROI-based lane violation detection + CSV logging
  detector.py           Standalone detection module

utils/
  video.py              Video I/O (file, webcam, YouTube)
  calibration.py        Interactive line & ROI drawing
  drawing.py            OpenCV overlay rendering

configs/
  settings.yaml         All tunable parameters
```

## Installation

**Prerequisites:** Python 3.10+, NVIDIA GPU with CUDA 12.4 (recommended)

```bash
git clone https://github.com/berkc3/vision-track.git
cd vision-track
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

> YOLO model weights are downloaded automatically on first run.

## Usage

### CLI

```bash
# Local video file
python main.py --source road.mp4

# Webcam
python main.py --source 0

# YouTube stream
python main.py --source "https://www.youtube.com/watch?v=..."

# Skip interactive calibration, save output
python main.py --source road.mp4 --no-calibrate --output output.mp4

# Headless mode (no display window)
python main.py --source road.mp4 --no-display --output output.mp4
```

### Streamlit Web UI

```bash
streamlit run ui/app.py
```

### Interactive Calibration

On startup (unless `--no-calibrate`), the first stable frame (after skipping initial seconds) is shown for calibration:

**Speed Path Calibration:**
- Left-click to draw Path 1 (start line) - green
- Right-click to draw Path 2 (end line) - orange
- Press `c` to confirm, `r` to reset, `q` to skip

**ROI Calibration (Restricted Zones):**
- Left-click to add polygon points
- Right-click to close current polygon
- Press `n` for new zone, `c` to confirm all, `q` to skip
- Zones are labeled **Restricted Zone 1**, **Restricted Zone 2**, … in the live video output

## Configuration

All parameters are in [`configs/settings.yaml`](configs/settings.yaml):

```yaml
model:
  path: "yolo11x.pt"           # see Model Selection Guide below
  confidence: 0.4
  imgsz: 1920                  # inference resolution

speed:
  real_distance_m: 15.0        # actual distance between speed lines (meters)
  calibration_mode: true
  calibration_skip_seconds: 5  # skip N seconds before grabbing calibration frame

violation:
  roi_polygon: []              # set via UI or define here

display:
  show_trails: true
  show_speed: true
  show_roi: true
  max_w: 1920                  # max display width for calibration and live windows
  max_h: 1080                  # max display height for calibration and live windows

output:
  save_violations: true        # saves outputs/violations_<timestamp>.csv
  output_dir: "outputs"
```

### Model Selection Guide

The system supports any Ultralytics YOLO model — both the **YOLOv8** and **YOLO11** families. Set `model.path` in `configs/settings.yaml` to the desired weights file. Weights are downloaded automatically on first use.

**YOLO11** (current generation, recommended):

| Model | Speed (RTX 4080 Super @ 1920px) | mAP50-95 | Best For |
|-------|----------------------------------|----------|----------|
| `yolo11n.pt` | ~55 fps | 39.5 | Real-time, low-power |
| `yolo11s.pt` | ~40 fps | 47.0 | Balanced |
| `yolo11m.pt` | ~25 fps | 51.5 | General use |
| `yolo11l.pt` | ~18 fps | 53.4 | High accuracy |
| `yolo11x.pt` | ~12 fps | 54.7 | Traffic cameras, accuracy-first |

**YOLOv8** (previous generation, also supported):

| Model | Speed (RTX 4080 Super @ 1920px) | mAP50-95 | Best For |
|-------|----------------------------------|----------|----------|
| `yolov8n.pt` | ~60 fps | 37.3 | Real-time, low-power |
| `yolov8s.pt` | ~45 fps | 44.9 | Balanced |
| `yolov8m.pt` | ~28 fps | 50.2 | General use |
| `yolov8l.pt` | ~20 fps | 52.9 | High accuracy |
| `yolov8x.pt` | ~14 fps | 53.9 | Traffic cameras, accuracy-first |

> YOLO11 achieves higher mAP with ~17% fewer parameters than its YOLOv8 counterpart. For new deployments, YOLO11 is recommended. YOLOv8 remains a valid and well-tested alternative.

## How It Works

1. **Detection & Tracking** - Each frame is passed through an Ultralytics YOLO model with BoT-SORT tracking (`persist=True`). Vehicles get unique IDs that persist across frames.

2. **Speed Estimation** - Two polyline paths are placed across the road. When a vehicle's center crosses Path 1, a timer starts. When it crosses Path 2, the timer stops. Speed = `real_distance / time_elapsed`.

3. **Violation Detection** - ROI polygons define restricted zones labeled **Restricted Zone 1**, **Restricted Zone 2**, etc. `cv2.pointPolygonTest()` checks if a vehicle center is inside any zone. Each `(vehicle_id, zone)` pair is recorded **exactly once** — no duplicate alerts — and appended to a timestamped CSV file in `outputs/`.

4. **Rendering** - Bounding boxes, trail polylines, speed labels, violation flashes, and a stats panel are composited onto each frame using OpenCV's drawing and alpha blending functions.

## Violation Log

Each run with `save_violations: true` produces a CSV in `outputs/violations_<timestamp>.csv`:

```
timestamp,track_id,class_name,roi_zone,center_x,center_y
2026-02-19 14:30:15,10,car,1,854.3,612.7
2026-02-19 14:30:22,10,car,2,921.1,589.4
2026-02-19 14:30:31,15,truck,1,743.8,634.2
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | Ultralytics YOLO (YOLOv8 / YOLO11) | Real-time object detection |
| Tracking | BoT-SORT | Multi-object tracking with re-ID |
| Backend | PyTorch + CUDA | GPU-accelerated inference |
| Vision | OpenCV | Video I/O, rendering, geometry |
| Web UI | Streamlit | Browser-based dashboard |
| Streaming | yt-dlp | YouTube URL resolution |
| Config | PyYAML | Human-readable configuration |

## License

MIT
