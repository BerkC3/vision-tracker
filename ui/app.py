from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tracker import VehicleTracker
from core.speed_estimator import SpeedEstimator
from core.violation import LaneViolationDetector
from utils.drawing import OverlayRenderer

st.set_page_config(page_title="Vision-Track AI", page_icon="🚗", layout="wide")


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_tracker(model_path: str, confidence: float, device: str, classes: list[int], imgsz: int) -> VehicleTracker:
    return VehicleTracker(
        model_path=model_path,
        confidence=confidence,
        device=device,
        classes=classes,
        imgsz=imgsz,
    )


def main() -> None:
    st.title("🚗 Vision-Track AI")
    st.caption("Real-time Traffic Analysis with Ultralytics YOLO + BoT-SORT")

    config = load_config()

    with st.sidebar:
        st.header("⚙️ Settings")

        source_type = st.radio("Video Source", ["Upload", "Webcam", "YouTube URL"])

        video_file = None
        video_source = None

        if source_type == "Upload":
            video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        elif source_type == "Webcam":
            video_source = 0
        else:
            video_source = st.text_input("YouTube URL")

        st.divider()
        st.subheader("Detection")
        confidence = st.slider("Confidence", 0.1, 1.0, config["model"]["confidence"], 0.05)
        model_size = st.selectbox(
            "Model",
            [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            ],
            help="YOLO11 (recommended) or YOLOv8. Larger models are more accurate but slower.",
        )
        imgsz = st.selectbox("Inference Resolution", [640, 1280, 1920], index=2)

        st.divider()
        st.subheader("Speed Lines")
        st.info("In CLI mode, lines are drawn interactively on the video. Here, use Y-ratio sliders as horizontal lines.")
        line1_ratio = st.slider("Line 1 Y-Position", 0.1, 0.9, config["speed"]["line1_y_ratio"], 0.05)
        line2_ratio = st.slider("Line 2 Y-Position", 0.1, 0.9, config["speed"]["line2_y_ratio"], 0.05)
        real_distance = st.number_input("Distance between lines (m)", 1.0, 100.0, config["speed"]["real_distance_m"])

        st.divider()
        st.subheader("ROI (Lane Violation)")
        roi_mode = st.radio("ROI Mode", ["None", "Manual Coordinates"])
        roi_polygon = None
        if roi_mode == "Manual Coordinates":
            roi_text = st.text_area(
                "Polygon points (x,y per line)",
                placeholder="100,200\n300,200\n300,500\n100,500",
            )
            if roi_text.strip():
                try:
                    roi_polygon = [
                        [int(c) for c in line.split(",")]
                        for line in roi_text.strip().splitlines()
                        if "," in line
                    ]
                except ValueError:
                    st.error("Invalid coordinates. Use format: x,y")

        st.divider()
        st.subheader("Display")
        show_trails = st.checkbox("Show trails", config["display"]["show_trails"])
        show_speed = st.checkbox("Show speed", config["display"]["show_speed"])
        show_roi = st.checkbox("Show ROI", config["display"]["show_roi"])
        stats_opacity = st.slider("Stats panel opacity", 0.1, 0.8, config["display"].get("stats_opacity", 0.35), 0.05)

    col_video, col_stats = st.columns([3, 1])

    with col_stats:
        st.subheader("📊 Live Stats")
        total_placeholder = st.empty()
        speed_placeholder = st.empty()
        violation_placeholder = st.empty()
        fps_placeholder = st.empty()
        st.divider()
        progress_placeholder = st.empty()

    with col_video:
        frame_placeholder = st.empty()

    start_btn = st.sidebar.button("▶ Start Processing", type="primary", use_container_width=True)

    if not start_btn:
        st.info("Upload a video or select a source, then click **Start Processing**.")
        return

    cap = None
    if video_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
    elif video_source == 0:
        cap = cv2.VideoCapture(0)
    elif isinstance(video_source, str) and video_source:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]", "-g", video_source],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            cap = cv2.VideoCapture(result.stdout.strip())
        else:
            st.error(f"yt-dlp failed: {result.stderr}")
            return

    if cap is None or not cap.isOpened():
        st.error("Could not open video source.")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    path1 = [(0, int(h * line1_ratio)), (w, int(h * line1_ratio))]
    path2 = [(0, int(h * line2_ratio)), (w, int(h * line2_ratio))]

    tracker = get_tracker(model_size, confidence, config["model"]["device"], config["model"]["classes"], imgsz)
    tracker.reset()

    speed_est = SpeedEstimator(path1, path2, real_distance)
    roi_polygons = [roi_polygon] if roi_polygon else None
    violation_det = LaneViolationDetector(roi_polygons=roi_polygons)

    renderer = OverlayRenderer(
        show_trails=show_trails,
        show_speed=show_speed,
        show_roi=show_roi,
        stats_opacity=stats_opacity,
    )

    frame_count = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    t_start = time.monotonic()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_timestamp = frame_count / video_fps
        vehicles = tracker.update(frame)

        renderer.draw_speed_paths(frame, path1, path2)
        if violation_det.has_roi:
            for i, poly in enumerate(violation_det.polygons):
                renderer.draw_roi(frame, poly, label=f"Restricted Zone {i + 1}")

        for v in vehicles:
            speed = speed_est.estimate(v.track_id, v.center, timestamp=video_timestamp)
            violation = violation_det.check(v.track_id, v.center, v.class_name, frame)
            renderer.draw_vehicle(
                frame, v.bbox, v.track_id, v.class_name, v.confidence,
                speed=speed, trail=v.trail, is_violating=violation is not None,
            )

        frame_count += 1
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0

        renderer.draw_stats(
            frame,
            total_vehicles=tracker.total_unique_vehicles,
            avg_speed=speed_est.average_speed,
            violations=violation_det.violation_count,
            fps=fps,
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        total_placeholder.metric("Total Vehicles", tracker.total_unique_vehicles)
        avg_spd = speed_est.average_speed
        speed_placeholder.metric("Avg Speed", f"{avg_spd:.1f} km/h" if avg_spd else "--")
        violation_placeholder.metric("Violations", violation_det.violation_count)
        fps_placeholder.metric("FPS", f"{fps:.1f}")

        if total_frames > 0:
            progress_placeholder.progress(frame_count / total_frames, text=f"Frame {frame_count}/{total_frames}")

    cap.release()
    st.success(f"Processing complete! {frame_count} frames | {tracker.total_unique_vehicles} vehicles detected")


if __name__ == "__main__":
    main()
