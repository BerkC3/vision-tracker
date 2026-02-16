from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViolationEvent:
    track_id: int
    class_name: str
    center: tuple[float, float]
    timestamp: float
    roi_index: int = 0
    frame_snapshot: np.ndarray | None = None


class LaneViolationDetector:
    def __init__(
        self,
        roi_polygons: list[list[list[int]]] | None = None,
        cooldown_seconds: float = 3.0,
        save_snapshots: bool = True,
    ) -> None:
        self._polygons: list[np.ndarray] = []
        if roi_polygons:
            for poly in roi_polygons:
                self.add_roi(poly)
        self.cooldown_seconds = cooldown_seconds
        self.save_snapshots = save_snapshots
        self._last_violation: dict[int, float] = {}
        self._violations: list[ViolationEvent] = []

    def add_roi(self, polygon: list[list[int]]) -> None:
        self._polygons.append(np.array(polygon, dtype=np.int32))
        logger.info(f"ROI #{len(self._polygons)} set with {len(polygon)} points")

    @property
    def has_roi(self) -> bool:
        return len(self._polygons) > 0

    @property
    def polygons(self) -> list[np.ndarray]:
        return self._polygons

    @property
    def violations(self) -> list[ViolationEvent]:
        return self._violations

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def check(
        self,
        track_id: int,
        center: tuple[float, float],
        class_name: str,
        frame: np.ndarray | None = None,
    ) -> ViolationEvent | None:
        if not self._polygons:
            return None

        point = (int(center[0]), int(center[1]))

        hit_index = -1
        for i, poly in enumerate(self._polygons):
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                hit_index = i
                break

        if hit_index < 0:
            return None

        now = time.monotonic()
        last = self._last_violation.get(track_id, 0)
        if now - last < self.cooldown_seconds:
            return None

        self._last_violation[track_id] = now

        snapshot = None
        if self.save_snapshots and frame is not None:
            snapshot = frame.copy()

        event = ViolationEvent(
            track_id=track_id,
            class_name=class_name,
            center=center,
            timestamp=now,
            roi_index=hit_index,
            frame_snapshot=snapshot,
        )
        self._violations.append(event)
        logger.warning(f"VIOLATION: Vehicle #{track_id} ({class_name}) in restricted zone #{hit_index + 1}")
        return event

    def reset(self) -> None:
        self._last_violation.clear()
        self._violations.clear()
