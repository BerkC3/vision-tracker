from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import cv2
import yaml

from core.tracker import VehicleTracker
from core.speed_estimator import SpeedEstimator
from core.violation import LaneViolationDetector
from utils.drawing import OverlayRenderer
from utils.calibration import PathCalibrator, ROICalibrator
from utils.video import VideoSource, VideoWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision-track")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _default_paths(h: int, w: int, config: dict) -> tuple[list, list]:
    """Fallback horizontal paths from config ratios."""
    y1 = int(h * config["speed"]["line1_y_ratio"])
    y2 = int(h * config["speed"]["line2_y_ratio"])
    return [(0, y1), (w, y1)], [(0, y2), (w, y2)]


def run(source: str | int, config: dict, output_path: str | None = None, show: bool = True, test_mode: bool = False) -> None:
    model_cfg = config["model"]
    speed_cfg = config["speed"]
    viol_cfg = config["violation"]
    disp_cfg = config["display"]

    tracker = VehicleTracker(
        model_path=model_cfg["path"],
        confidence=model_cfg["confidence"],
        iou_threshold=model_cfg["iou_threshold"],
        device=model_cfg["device"],
        classes=model_cfg["classes"],
        trail_length=disp_cfg["trail_length"],
        imgsz=model_cfg.get("imgsz", 1920),
    )

    video = VideoSource(source)
    video.open()
    h, w = video.height, video.width

    # Grab a stable frame for calibration (skip initial seconds to avoid camera shake)
    skip_seconds = speed_cfg.get("calibration_skip_seconds", 0)
    skip_frames = int(skip_seconds * (video.fps or 25.0))
    calib_frame = None
    for i, frame in enumerate(video.frames()):
        if i >= skip_frames:
            calib_frame = frame
            break

    if calib_frame is None:
        logger.error("Cannot read calibration frame.")
        return

    max_w = disp_cfg.get("max_w", 1920)
    max_h = disp_cfg.get("max_h", 1080)

    # Speed paths calibration
    if test_mode:
        path1 = [(0, int(h * 0.4)), (w, int(h * 0.4))]
        path2 = [(0, int(h * 0.65)), (w, int(h * 0.65))]
        logger.info(f"TEST MODE: Fixed speed paths at y={int(h * 0.4)} and y={int(h * 0.65)}")
    else:
        path1, path2 = _default_paths(h, w, config)
        if speed_cfg.get("calibration_mode", False) and show:
            calibrator = PathCalibrator()
            result = calibrator.calibrate(calib_frame, max_w=max_w, max_h=max_h)
            if result is not None:
                path1, path2 = result
                logger.info("Using user-drawn speed paths")
            else:
                logger.info("Using default speed paths from config")

    speed_est = SpeedEstimator(
        path1=path1,
        path2=path2,
        real_distance_m=speed_cfg["real_distance_m"],
        smoothing_window=speed_cfg["smoothing_window"],
    )

    # ROI calibration (multiple zones)
    roi_polygons = None
    cfg_roi = viol_cfg.get("roi_polygon") or []
    if cfg_roi:
        # Config has a single polygon - wrap in list
        roi_polygons = [cfg_roi] if cfg_roi and isinstance(cfg_roi[0], list) and not isinstance(cfg_roi[0][0], list) else cfg_roi
    if not roi_polygons and show and not test_mode:
        roi_cal = ROICalibrator()
        roi_polygons = roi_cal.calibrate(calib_frame, max_w=max_w, max_h=max_h)

    out_cfg = config.get("output", {})
    violations_file = None
    if out_cfg.get("save_violations", True):
        out_dir = out_cfg.get("output_dir", "outputs")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_file = f"{out_dir}/violations_{ts}.csv"

    violation_det = LaneViolationDetector(
        roi_polygons=roi_polygons,
        violations_file=violations_file,
    )

    renderer = OverlayRenderer(
        box_thickness=disp_cfg["box_thickness"],
        font_scale=disp_cfg["font_scale"],
        show_trails=disp_cfg["show_trails"],
        show_speed=disp_cfg["show_speed"],
        show_roi=disp_cfg["show_roi"],
        stats_opacity=disp_cfg.get("stats_opacity", 0.35),
    )

    writer = None
    if output_path:
        writer = VideoWriter(output_path, video.fps, w, h)

    cv2.namedWindow("Vision-Track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision-Track", min(w, max_w), min(h, max_h))
    logger.info("Pipeline started. Press 'q' to quit.")
    frame_count = 0
    video_fps = video.fps or 25.0
    t_start = time.monotonic()

    # Re-open to process from beginning
    video.release()
    video = VideoSource(source)
    video.open()

    try:
        for frame in video.frames():
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

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Vision-Track", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User quit.")
                    break

    finally:
        video.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(
            f"Done. {frame_count} frames | "
            f"{tracker.total_unique_vehicles} vehicles | "
            f"{violation_det.violation_count} violations"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-Track: Traffic Analysis System")
    parser.add_argument("--source", type=str, default="0", help="Video path, webcam index, or YouTube URL")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable window display")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip interactive calibration")
    parser.add_argument("--test", action="store_true", help="Use fixed speed paths for consistent testing")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.no_calibrate or args.test:
        config["speed"]["calibration_mode"] = False

    source: str | int = args.source
    if source.isdigit():
        source = int(source)

    run(source=source, config=config, output_path=args.output, show=not args.no_display, test_mode=args.test)


if __name__ == "__main__":
    main()
