"""
tests/test_violation.py
=======================
Unit tests for core.violation.LaneViolationDetector.

All tests use use_database=False so they run without touching SQLite,
making them fast and dependency-free.  The one DB-related test mocks
get_session() to verify that write errors are silently caught.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import core.database as db_mod
from core.database import Base
from core.violation import LaneViolationDetector, ViolationEvent

# A simple 80x80 square polygon used across multiple tests.
SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]

# A second polygon in a different area of the frame.
SQUARE2 = [[200, 200], [280, 200], [280, 280], [200, 280]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> LaneViolationDetector:
    """Detector with a single ROI square and no database."""
    return LaneViolationDetector(roi_polygons=[SQUARE], use_database=False)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 300x300 black frame for snapshot tests."""
    return np.zeros((300, 300, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_no_roi_by_default(self) -> None:
        d = LaneViolationDetector(use_database=False)
        assert not d.has_roi
        assert d.violation_count == 0
        assert d.violations == []

    def test_roi_polygons_loaded_at_init(self) -> None:
        d = LaneViolationDetector(roi_polygons=[SQUARE], use_database=False)
        assert d.has_roi
        assert len(d.polygons) == 1

    def test_multiple_roi_polygons_at_init(self) -> None:
        d = LaneViolationDetector(roi_polygons=[SQUARE, SQUARE2], use_database=False)
        assert len(d.polygons) == 2


# ---------------------------------------------------------------------------
# add_roi
# ---------------------------------------------------------------------------


class TestAddRoi:
    def test_add_roi_increments_polygon_count(self) -> None:
        d = LaneViolationDetector(use_database=False)
        assert len(d.polygons) == 0
        d.add_roi(SQUARE)
        assert len(d.polygons) == 1
        d.add_roi(SQUARE2)
        assert len(d.polygons) == 2

    def test_add_roi_stores_numpy_array(self) -> None:
        d = LaneViolationDetector(use_database=False)
        d.add_roi(SQUARE)
        assert isinstance(d.polygons[0], np.ndarray)
        assert d.polygons[0].dtype == np.int32

    def test_has_roi_true_after_add(self) -> None:
        d = LaneViolationDetector(use_database=False)
        assert not d.has_roi
        d.add_roi(SQUARE)
        assert d.has_roi


# ---------------------------------------------------------------------------
# check — detection logic
# ---------------------------------------------------------------------------


class TestCheck:
    def test_no_roi_always_returns_none(self) -> None:
        d = LaneViolationDetector(use_database=False)
        assert d.check(1, (50.0, 50.0), "car") is None

    def test_point_outside_polygon_returns_none(self, detector) -> None:
        # (200, 200) is outside the [10-90] square
        assert detector.check(1, (200.0, 200.0), "car") is None

    def test_point_inside_polygon_returns_event(self, detector) -> None:
        event = detector.check(1, (50.0, 50.0), "car")
        assert event is not None
        assert isinstance(event, ViolationEvent)

    def test_returned_event_has_correct_fields(self, detector) -> None:
        event = detector.check(42, (50.0, 50.0), "truck")
        assert event.track_id == 42
        assert event.class_name == "truck"
        assert event.roi_index == 0
        assert isinstance(event.timestamp, datetime)
        assert event.center == (50.0, 50.0)

    def test_violation_count_increments(self, detector) -> None:
        detector.check(1, (50.0, 50.0), "car")
        detector.check(2, (50.0, 50.0), "truck")
        assert detector.violation_count == 2

    def test_violations_list_populated(self, detector) -> None:
        detector.check(1, (50.0, 50.0), "car")
        assert len(detector.violations) == 1
        assert detector.violations[0].track_id == 1

    # ------------------------------------------------------------------
    # Once-per-(track_id, zone) guarantee
    # ------------------------------------------------------------------

    def test_same_vehicle_same_zone_fires_once(self, detector) -> None:
        """The second entry of the same vehicle into the same zone is ignored."""
        r1 = detector.check(1, (50.0, 50.0), "car")
        r2 = detector.check(1, (55.0, 55.0), "car")
        assert r1 is not None
        assert r2 is None
        assert detector.violation_count == 1

    def test_different_vehicles_both_fire(self, detector) -> None:
        r1 = detector.check(1, (50.0, 50.0), "car")
        r2 = detector.check(2, (50.0, 50.0), "truck")
        assert r1 is not None and r2 is not None
        assert detector.violation_count == 2

    def test_same_vehicle_different_zones_both_fire(self) -> None:
        """One vehicle entering two distinct zones must produce two events."""
        d = LaneViolationDetector(roi_polygons=[SQUARE, SQUARE2], use_database=False)
        r1 = d.check(1, (50.0, 50.0), "car")  # inside SQUARE (zone 0)
        r2 = d.check(1, (240.0, 240.0), "car")  # inside SQUARE2 (zone 1)
        assert r1 is not None and r2 is not None
        assert r1.roi_index == 0
        assert r2.roi_index == 1
        assert d.violation_count == 2

    # ------------------------------------------------------------------
    # Snapshot behaviour
    # ------------------------------------------------------------------

    def test_snapshot_captured_when_enabled(self, detector, blank_frame) -> None:
        event = detector.check(1, (50.0, 50.0), "car", frame=blank_frame)
        assert event is not None
        assert event.frame_snapshot is not None
        assert event.frame_snapshot.shape == blank_frame.shape

    def test_snapshot_is_a_copy(self, detector, blank_frame) -> None:
        """Mutating the original frame must not affect the stored snapshot."""
        event = detector.check(1, (50.0, 50.0), "car", frame=blank_frame)
        blank_frame[:] = 255  # modify original
        assert event.frame_snapshot.sum() == 0  # snapshot unchanged

    def test_no_snapshot_when_disabled(self, blank_frame) -> None:
        d = LaneViolationDetector(
            roi_polygons=[SQUARE], save_snapshots=False, use_database=False
        )
        event = d.check(1, (50.0, 50.0), "car", frame=blank_frame)
        assert event is not None
        assert event.frame_snapshot is None

    def test_no_snapshot_when_no_frame_provided(self, detector) -> None:
        event = detector.check(1, (50.0, 50.0), "car", frame=None)
        assert event is not None
        assert event.frame_snapshot is None

    # ------------------------------------------------------------------
    # Point exactly on polygon boundary
    # ------------------------------------------------------------------

    def test_point_on_boundary_is_inside(self, detector) -> None:
        """cv2.pointPolygonTest returns 0 for boundary points — treated as inside."""
        # Top-left corner of the square
        event = detector.check(1, (10.0, 10.0), "car")
        assert event is not None


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_violation_list(self, detector) -> None:
        detector.check(1, (50.0, 50.0), "car")
        detector.reset()
        assert detector.violation_count == 0
        assert detector.violations == []

    def test_reset_clears_seen_track_ids(self, detector) -> None:
        """After reset, the same (track_id, zone) pair can trigger again."""
        detector.check(1, (50.0, 50.0), "car")
        detector.reset()
        event = detector.check(1, (50.0, 50.0), "car")
        assert event is not None

    def test_reset_does_not_remove_polygons(self, detector) -> None:
        detector.reset()
        assert detector.has_roi
        assert len(detector.polygons) == 1


# ---------------------------------------------------------------------------
# Database error isolation
# ---------------------------------------------------------------------------


class TestDatabaseErrorIsolation:
    def test_db_write_error_does_not_crash_pipeline(self) -> None:
        """A failure inside get_session() must be caught; the in-memory
        ViolationEvent must still be recorded and returned."""
        d = LaneViolationDetector(roi_polygons=[SQUARE], use_database=True)

        with patch("core.violation.get_session", side_effect=RuntimeError("DB down")):
            event = d.check(99, (50.0, 50.0), "car")

        # Detection must succeed even though the DB write failed.
        assert event is not None
        assert event.track_id == 99
        assert d.violation_count == 1

    def test_db_write_succeeds_with_real_session(self, monkeypatch) -> None:
        """When a real (in-memory) session is wired up, _save_to_db must
        persist a ViolationRecord — covering the happy-path body of the method
        (lines 94-102 in violation.py)."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        monkeypatch.setattr(db_mod, "_engine", engine)
        monkeypatch.setattr(db_mod, "_SessionFactory", factory)

        d = LaneViolationDetector(roi_polygons=[SQUARE], use_database=True)
        event = d.check(77, (50.0, 50.0), "bus")

        assert event is not None
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM violations WHERE track_id=77")
            ).scalar()
        assert count == 1
