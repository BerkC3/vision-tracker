from __future__ import annotations

import cv2
import numpy as np

# Color palette for different vehicle classes
CLASS_COLORS = {
    "car": (0, 255, 0),        # green
    "truck": (255, 165, 0),    # orange
    "bus": (255, 0, 255),      # magenta
    "motorcycle": (0, 255, 255), # yellow
    "unknown": (200, 200, 200),
}

VIOLATION_COLOR = (0, 0, 255)  # red
LINE_COLOR = (0, 200, 255)    # cyan-yellow
ROI_COLOR = (255, 100, 0)     # blue-ish


class OverlayRenderer:
    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        show_trails: bool = True,
        show_speed: bool = True,
        show_roi: bool = True,
        stats_opacity: float = 0.35,
    ) -> None:
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_trails = show_trails
        self.show_speed = show_speed
        self.show_roi = show_roi
        self.stats_opacity = stats_opacity
        self._violation_flash: dict[int, int] = {}

    def draw_vehicle(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        track_id: int,
        class_name: str,
        confidence: float,
        speed: float | None = None,
        trail: list[tuple[float, float]] | None = None,
        is_violating: bool = False,
    ) -> None:
        x1, y1, x2, y2 = map(int, bbox)
        color = VIOLATION_COLOR if is_violating else CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # Label background
        label = f"#{track_id} {class_name}"
        if self.show_speed and speed is not None:
            label += f" {speed:.0f}km/h"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 2)

        # Trail
        if self.show_trails and trail and len(trail) > 1:
            pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 2)

        # Violation flash effect
        if is_violating:
            self._violation_flash[track_id] = 10
        remaining = self._violation_flash.get(track_id, 0)
        if remaining > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), VIOLATION_COLOR, -1)
            alpha = remaining / 20.0
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            self._violation_flash[track_id] = remaining - 1

    def draw_speed_paths(
        self,
        frame: np.ndarray,
        path1: list[tuple[int, int]],
        path2: list[tuple[int, int]],
    ) -> None:
        if path1:
            pts = np.array(path1, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
            for p in path1:
                cv2.circle(frame, p, 4, (0, 255, 0), -1)
            cv2.putText(frame, "PATH 1", (path1[0][0] + 8, path1[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if path2:
            pts = np.array(path2, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 140, 255), 3, cv2.LINE_AA)
            for p in path2:
                cv2.circle(frame, p, 4, (0, 140, 255), -1)
            cv2.putText(frame, "PATH 2", (path2[0][0] + 8, path2[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    def draw_roi(self, frame: np.ndarray, polygon: np.ndarray) -> None:
        if not self.show_roi or polygon is None:
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (*ROI_COLOR, 50))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [polygon], True, ROI_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "RESTRICTED ZONE", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)

    def draw_stats(
        self,
        frame: np.ndarray,
        total_vehicles: int,
        avg_speed: float | None,
        violations: int,
        fps: float,
    ) -> None:
        h, w = frame.shape[:2]
        panel_w, panel_h = 260, 130
        x, y = w - panel_w - 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.stats_opacity, frame, 1 - self.stats_opacity, 0, frame)

        lines = [
            f"Vehicles: {total_vehicles}",
            f"Avg Speed: {avg_speed:.1f} km/h" if avg_speed else "Avg Speed: --",
            f"Violations: {violations}",
            f"FPS: {fps:.1f}",
        ]
        for i, text in enumerate(lines):
            color = VIOLATION_COLOR if "Violations" in text and violations > 0 else (255, 255, 255)
            cv2.putText(frame, text, (x + 10, y + 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
