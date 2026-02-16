# Vision-Track

Real-time traffic analysis system powered by YOLOv8 and BoT-SORT. Detects vehicles, tracks them across frames, estimates speed, and flags lane violations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Vehicle Detection** - YOLOv8 (nano to extra-large) with COCO vehicle classes (car, motorcycle, bus, truck)
- **Multi-Object Tracking** - BoT-SORT with re-identification for robust tracking through occlusions
- **Speed Estimation** - Two-line crossing method with perspective-aware polyline paths
- **Lane Violation Detection** - Multiple ROI polygon zones with cooldown-based alerting
- **Interactive Calibration** - Draw speed lines and restricted zones directly on the first frame
- **Dual Interface** - CLI with OpenCV display + Streamlit web dashboard
- **Flexible Input** - Local video files, webcam, or YouTube URLs (via yt-dlp)

## Architecture

```
main.py                 CLI entry point
ui/app.py               Streamlit web UI

core/
  tracker.py            YOLOv8 + BoT-SORT detection & tracking
  speed_estimator.py    Two-line speed measurement
  violation.py          ROI-based lane violation detection
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
git clone https://github.com/<berkc3>/vision-track.git
cd vision-track
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

> YOLOv8 model weights are downloaded automatically on first run.

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

On startup (unless `--no-calibrate`), two calibration windows appear:

**Speed Path Calibration:**
- Left-click to draw Path 1 (start line) - green
- Right-click to draw Path 2 (end line) - orange
- Press `c` to confirm, `r` to reset, `q` to skip

**ROI Calibration (Restricted Zones):**
- Left-click to add polygon points
- Right-click to close current polygon
- Press `n` for new zone, `c` to confirm all, `q` to skip

## Configuration

All parameters are in [`configs/settings.yaml`](configs/settings.yaml):

```yaml
model:
  path: "yolov8x.pt"          # yolov8n/s/m/l/x
  confidence: 0.4
  imgsz: 1920                 # inference resolution
  classes: [2, 3, 5, 7]       # car, motorcycle, bus, truck

speed:
  real_distance_m: 15.0        # actual distance between speed lines (meters)
  calibration_mode: true

violation:
  cooldown_seconds: 3.0

display:
  show_trails: true
  show_speed: true
  show_roi: true
```

### Model Selection Guide

| Model | Speed (RTX 4080 Super) | Accuracy | Best For |
|-------|------------------------|----------|----------|
| `yolov8n.pt` | ~60 fps | Good | Real-time, low-power |
| `yolov8s.pt` | ~45 fps | Better | Balanced |
| `yolov8m.pt` | ~30 fps | High | General use |
| `yolov8x.pt` | ~15 fps | Highest | Traffic cameras, accuracy-first |

## How It Works

1. **Detection & Tracking** - Each frame is passed through YOLOv8 with BoT-SORT tracking (`persist=True`). Vehicles get unique IDs that persist across frames.

2. **Speed Estimation** - Two polyline paths are placed across the road. When a vehicle's center crosses Path 1, a timer starts. When it crosses Path 2, the timer stops. Speed = `real_distance / time_elapsed`.

3. **Violation Detection** - ROI polygons define restricted zones. `cv2.pointPolygonTest()` checks if a vehicle center is inside any zone. A cooldown prevents duplicate alerts for the same vehicle.

4. **Rendering** - Bounding boxes, trail polylines, speed labels, violation flashes, and a stats panel are composited onto each frame using OpenCV's drawing and alpha blending functions.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLOv8 (Ultralytics) | Real-time object detection |
| Tracking | BoT-SORT | Multi-object tracking with re-ID |
| Backend | PyTorch + CUDA | GPU-accelerated inference |
| Vision | OpenCV | Video I/O, rendering, geometry |
| Web UI | Streamlit | Browser-based dashboard |
| Streaming | yt-dlp | YouTube URL resolution |
| Config | PyYAML | Human-readable configuration |

## License

MIT
