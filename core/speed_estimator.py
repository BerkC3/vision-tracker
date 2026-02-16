from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _point_to_polyline_distance(point: tuple[float, float], polyline: list[tuple[int, int]]) -> float:
    """Minimum distance from a point to any segment of the polyline."""
    min_dist = float("inf")
    px, py = point
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_sq = abx * abx + aby * aby

        if ab_sq == 0:
            dist = (apx * apx + apy * apy) ** 0.5
        else:
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

        min_dist = min(min_dist, dist)
    return min_dist


def _crossed_polyline(
    polyline: list[tuple[int, int]],
    prev: tuple[float, float],
    curr: tuple[float, float],
) -> bool:
    """Check if movement from prev to curr crosses any segment of the polyline."""
    for i in range(len(polyline) - 1):
        if _segments_intersect(prev, curr, polyline[i], polyline[i + 1]):
            return True
    return False


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[int, int], p4: tuple[int, int],
) -> bool:
    """Check if line segment p1-p2 intersects segment p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True

    return False


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and \
           min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


@dataclass
class SpeedRecord:
    track_id: int
    speed_kmh: float
    timestamp: float


class SpeedEstimator:
    """Speed estimation using two polyline paths.

    Vehicles crossing path1 start the timer, crossing path2 stops it.
    Works with perspective views where roads aren't straight horizontal lines.
    """

    def __init__(
        self,
        path1: list[tuple[int, int]],
        path2: list[tuple[int, int]],
        real_distance_m: float = 15.0,
        smoothing_window: int = 5,
    ) -> None:
        self.path1 = path1
        self.path2 = path2
        self.real_distance_m = real_distance_m
        self.smoothing_window = smoothing_window

        self._path1_times: dict[int, float] = {}
        self._path2_times: dict[int, float] = {}
        self._speeds: dict[int, list[float]] = defaultdict(list)
        self._prev_pos: dict[int, tuple[float, float]] = {}

    def estimate(self, track_id: int, center: tuple[float, float], timestamp: float = 0.0) -> float | None:
        """Estimate speed for a tracked vehicle.

        Args:
            track_id: Unique track ID.
            center: Vehicle center point (x, y).
            timestamp: Video timestamp in seconds (frame_index / fps).
        """
        prev = self._prev_pos.get(track_id)
        self._prev_pos[track_id] = center
        if prev is None:
            return self.get_speed(track_id)

        if _crossed_polyline(self.path1, prev, center):
            self._path1_times[track_id] = timestamp

        if _crossed_polyline(self.path2, prev, center):
            self._path2_times[track_id] = timestamp

        t1 = self._path1_times.get(track_id)
        t2 = self._path2_times.get(track_id)
        if t1 is not None and t2 is not None and t1 != t2:
            dt = abs(t2 - t1)
            speed_ms = self.real_distance_m / dt
            speed_kmh = speed_ms * 3.6

            if speed_kmh < 300:
                self._speeds[track_id].append(speed_kmh)
                if len(self._speeds[track_id]) > self.smoothing_window:
                    self._speeds[track_id] = self._speeds[track_id][-self.smoothing_window:]

            self._path1_times.pop(track_id, None)
            self._path2_times.pop(track_id, None)

        return self.get_speed(track_id)

    def get_speed(self, track_id: int) -> float | None:
        speeds = self._speeds.get(track_id)
        if not speeds:
            return None
        return sum(speeds) / len(speeds)

    @property
    def all_speeds(self) -> dict[int, float]:
        return {
            tid: sum(s) / len(s)
            for tid, s in self._speeds.items()
            if s
        }

    @property
    def average_speed(self) -> float | None:
        speeds = self.all_speeds
        if not speeds:
            return None
        return sum(speeds.values()) / len(speeds)

    def reset(self) -> None:
        self._path1_times.clear()
        self._path2_times.clear()
        self._speeds.clear()
        self._prev_pos.clear()
