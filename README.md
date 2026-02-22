# Vision-Track

Real-time traffic analysis system powered by Ultralytics YOLO and BoT-SORT. Detects vehicles, tracks them across frames, estimates speed, and flags lane violations with persistent database logging.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-red?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Vehicle Detection** - Ultralytics YOLO (YOLOv8 or YOLO11, nano to extra-large) with COCO vehicle classes (car, motorcycle, bus, truck)
- **Multi-Object Tracking** - BoT-SORT with re-identification for robust tracking through occlusions
- **Speed Estimation** - Two-line crossing method with perspective-aware polyline paths
- **Lane Violation Detection** - Multiple named ROI polygon zones; each (vehicle, zone) pair is recorded exactly once and persisted to SQLite or PostgreSQL
- **Interactive Calibration** - Draw speed lines and restricted zones on a configurable stable frame (skips initial camera shake)
- **Dual Interface** - CLI with OpenCV display + Streamlit web dashboard
- **Flexible Input** - Local video files, webcam, or YouTube URLs (via yt-dlp)
- **Database Persistence** - SQLAlchemy ORM; SQLite by default, switchable to PostgreSQL via environment variable
- **Docker Support** - Multi-stage Dockerfile + docker-compose for zero-setup deployment

## Architecture

```
main.py                 CLI entry point
ui/app.py               Streamlit web UI

core/
  tracker.py            YOLO + BoT-SORT detection & tracking
  speed_estimator.py    Two-line speed measurement
  violation.py          ROI-based lane violation detection + database logging
  detector.py           Standalone detection module
  database.py           SQLAlchemy ORM (SQLite / PostgreSQL)

utils/
  video.py              Video I/O (file, webcam, YouTube)
  calibration.py        Interactive line & ROI drawing
  drawing.py            OpenCV overlay rendering

configs/
  settings.yaml         All tunable parameters

tests/
  test_database.py      Unit tests — database layer (66 tests total)
  test_speed.py         Unit tests — speed estimator
  test_violation.py     Unit tests — violation detector

.github/workflows/
  ci.yml                GitHub Actions CI (lint + test)
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

## Docker

The easiest way to run Vision-Track — no Python setup required:

```bash
# Build and start (SQLite, CPU)
docker compose up --build

# Start in background
docker compose up -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

The Streamlit dashboard is available at `http://localhost:8501`.

**GPU support:** Uncomment the `deploy` section in `docker-compose.yml` and set `TORCH_INDEX_URL` to the CUDA 12.4 wheel index. Requires the NVIDIA Container Toolkit.

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
  path: "yolov8x.pt"           # see Model Selection Guide below
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
  save_violations: true        # enables database logging
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

3. **Violation Detection** - ROI polygons define restricted zones labeled **Restricted Zone 1**, **Restricted Zone 2**, etc. `cv2.pointPolygonTest()` checks if a vehicle center is inside any zone. Each `(vehicle_id, zone)` pair is recorded **exactly once** — no duplicate alerts — and persisted to the database.

4. **Rendering** - Bounding boxes, trail polylines, speed labels, violation flashes, and a stats panel are composited onto each frame using OpenCV's drawing and alpha blending functions.

## Database & Violation Log

Violations are persisted automatically via SQLAlchemy to `outputs/violations.db` (SQLite by default).

**Table: `violations`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `timestamp` | DATETIME | UTC datetime of violation |
| `track_id` | INTEGER | BoT-SORT track identifier |
| `class_name` | VARCHAR | Vehicle type (car, truck, bus, motorcycle) |
| `roi_zone` | INTEGER | 1-based restricted zone index |
| `center_x` | FLOAT | Vehicle center X coordinate (pixels) |
| `center_y` | FLOAT | Vehicle center Y coordinate (pixels) |

**Switching to PostgreSQL:**

```bash
# Set the environment variable before running
export DATABASE_URL=postgresql://user:password@localhost:5432/visiontrack
python main.py --source road.mp4
```

Or uncomment the `db` service in `docker-compose.yml` for a fully containerised setup.

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest tests/ --cov=core --cov-report=term-missing

# Run a specific test file
pytest tests/test_database.py -v
```

The test suite has **66 tests** covering the database layer, speed estimator, and violation detector. All tests run on CPU without a GPU.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | Ultralytics YOLO (YOLOv8 / YOLO11) | Real-time object detection |
| Tracking | BoT-SORT | Multi-object tracking with re-ID |
| Backend | PyTorch + CUDA | GPU-accelerated inference |
| Vision | OpenCV | Video I/O, rendering, geometry |
| Database | SQLAlchemy 2.0 + SQLite / PostgreSQL | Violation persistence |
| Web UI | Streamlit | Browser-based dashboard |
| Streaming | yt-dlp | YouTube URL resolution |
| Config | PyYAML | Human-readable configuration |
| Container | Docker + docker-compose | Zero-setup deployment |
| CI | GitHub Actions | Lint + test on every push |

## License

MIT
