"""
core/violation.py
=================
ROI-based lane-violation detection with persistent database logging.

CSV logging has been replaced by SQLAlchemy (see core/database.py).
Violations are written to the database configured via DATABASE_URL.
Set use_database=False to run without any persistence (e.g. quick tests).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from .database import ViolationRecord, get_session, init_db

logger = logging.getLogger(__name__)


@dataclass
class ViolationEvent:
    """In-memory representation of a single violation (not persisted directly).

    frame_snapshot holds a copy of the video frame at the moment of violation,
    used for snapshot saving if needed. It is NOT stored in the database.
    """

    track_id: int
    class_name: str
    center: tuple[float, float]
    timestamp: datetime
    roi_index: int = 0
    frame_snapshot: np.ndarray | None = None


class LaneViolationDetector:
    """Detect lane violations and persist events to the database.

    Parameters
    ----------
    roi_polygons:
        List of polygon definitions.  Each polygon is a list of [x, y] pairs.
    save_snapshots:
        Whether to keep a copy of the frame inside ViolationEvent in memory.
        Snapshots are not stored in the database.
    use_database:
        When True (default), violations are written to the database via
        SQLAlchemy.  Set False to disable all persistence (useful for tests
        or headless runs where no output directory exists).
    """

    def __init__(
        self,
        roi_polygons: list[list[list[int]]] | None = None,
        save_snapshots: bool = True,
        use_database: bool = True,
        max_events: int = 500,
    ) -> None:
        self._polygons: list[np.ndarray] = []
        if roi_polygons:
            for poly in roi_polygons:
                self.add_roi(poly)

        self.save_snapshots = save_snapshots
        self.use_database = use_database
        self._max_events = max_events

        # Permanent per-(track_id, roi_index) tracking — each pair fires exactly once.
        self._violated: dict[int, set[int]] = defaultdict(set)
        self._violations: list[ViolationEvent] = []
        self._total_count: int = 0

        # Ensure the violations table exists before the first write.
        if use_database:
            init_db()

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    def _save_to_db(self, event: ViolationEvent) -> None:
        """Write a ViolationEvent to the database inside a managed transaction.

        Errors are logged but do NOT propagate — a DB write failure must
        never crash the real-time tracking pipeline.
        """
        if not self.use_database:
            return
        try:
            with get_session() as session:
                record = ViolationRecord(
                    timestamp=event.timestamp,
                    track_id=event.track_id,
                    class_name=event.class_name,
                    roi_zone=event.roi_index + 1,  # convert to 1-based index
                    center_x=round(event.center[0], 1),
                    center_y=round(event.center[1], 1),
                )
                session.add(record)
        except Exception as exc:
            logger.error(
                "DB write failed for violation (track_id=%d): %s", event.track_id, exc
            )

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def add_roi(self, polygon: list[list[int]]) -> None:
        """Register a new restricted zone polygon."""
        self._polygons.append(np.array(polygon, dtype=np.int32))
        logger.info("ROI #%d set with %d points", len(self._polygons), len(polygon))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        return self._total_count

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def check(
        self,
        track_id: int,
        center: tuple[float, float],
        class_name: str,
        frame: np.ndarray | None = None,
    ) -> ViolationEvent | None:
        """Test whether a vehicle is inside a restricted zone.

        Returns a ViolationEvent on the first entry of a (track_id, zone)
        pair, or None if there is no new violation.

        Parameters
        ----------
        track_id:   BoT-SORT track identifier.
        center:     Vehicle bounding-box centre (x, y) in pixels.
        class_name: COCO class label string.
        frame:      Current video frame (used for optional snapshot).
        """
        if not self._polygons:
            return None

        point = (int(center[0]), int(center[1]))

        # Find which polygon (if any) contains the point.
        hit_index = -1
        for i, poly in enumerate(self._polygons):
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                hit_index = i
                break

        if hit_index < 0:
            return None

        # Each (track_id, roi_index) pair is recorded exactly once — no duplicates.
        if hit_index in self._violated[track_id]:
            return None
        self._violated[track_id].add(hit_index)

        snapshot = frame.copy() if self.save_snapshots and frame is not None else None

        event = ViolationEvent(
            track_id=track_id,
            class_name=class_name,
            center=center,
            timestamp=datetime.now(timezone.utc),
            roi_index=hit_index,
            frame_snapshot=snapshot,
        )
        self._violations.append(event)
        self._total_count += 1
        # Cap the in-memory list to prevent unbounded growth (each event may
        # hold a full-frame snapshot).  The database keeps all records.
        if len(self._violations) > self._max_events:
            self._violations.pop(0)
        self._save_to_db(event)

        logger.warning(
            "VIOLATION: Vehicle #%d (%s) entered Restricted Zone #%d",
            track_id,
            class_name,
            hit_index + 1,
        )
        return event

    def reset(self) -> None:
        """Clear all in-memory state (DB rows are unaffected)."""
        self._violated.clear()
        self._violations.clear()
        self._total_count = 0
