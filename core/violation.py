from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViolationEvent:
    track_id: int
    class_name: str
    center: tuple[float, float]
    timestamp: datetime
    roi_index: int = 0
    frame_snapshot: np.ndarray | None = None


_CSV_HEADER = ["timestamp", "track_id", "class_name", "roi_zone", "center_x", "center_y"]


class LaneViolationDetector:
    def __init__(
        self,
        roi_polygons: list[list[list[int]]] | None = None,
        save_snapshots: bool = True,
        violations_file: str | None = None,
    ) -> None:
        self._polygons: list[np.ndarray] = []
        if roi_polygons:
            for poly in roi_polygons:
                self.add_roi(poly)
        self.save_snapshots = save_snapshots
        # Permanent per-(track_id, roi_index) tracking — each pair fires exactly once.
        self._violated: dict[int, set[int]] = defaultdict(set)
        self._violations: list[ViolationEvent] = []
        self._violations_file: str | None = violations_file
        if violations_file:
            self._init_csv(violations_file)

    def _init_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADER)
        logger.info(f"Violations log: {path}")

    def _append_csv(self, event: ViolationEvent) -> None:
        if not self._violations_file:
            return
        with open(self._violations_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                event.track_id,
                event.class_name,
                event.roi_index + 1,
                f"{event.center[0]:.1f}",
                f"{event.center[1]:.1f}",
            ])

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

        # Each (track_id, roi_index) pair is recorded exactly once.
        if hit_index in self._violated[track_id]:
            return None
        self._violated[track_id].add(hit_index)

        snapshot = frame.copy() if self.save_snapshots and frame is not None else None

        event = ViolationEvent(
            track_id=track_id,
            class_name=class_name,
            center=center,
            timestamp=datetime.now(),
            roi_index=hit_index,
            frame_snapshot=snapshot,
        )
        self._violations.append(event)
        self._append_csv(event)
        logger.warning(
            f"VIOLATION: Vehicle #{track_id} ({class_name}) in restricted zone #{hit_index + 1}"
        )
        return event

    def reset(self) -> None:
        self._violated.clear()
        self._violations.clear()
