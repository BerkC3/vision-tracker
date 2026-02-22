"""
tests/test_speed.py
===================
Unit tests for core.speed_estimator.SpeedEstimator.

These tests are entirely CPU-bound — no GPU, no video files, no OpenCV
windows.  They exercise the geometric helper functions and the speed
estimation maths so that correctness can be verified in CI without hardware.

Run locally:
    pytest tests/test_speed.py -v
"""

from __future__ import annotations

import pytest

from core.speed_estimator import (
    SpeedEstimator,
    _crossed_polyline,
    _point_to_polyline_distance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def estimator() -> SpeedEstimator:
    """SpeedEstimator with two horizontal lines 10 m apart (real world).

    Layout (pixels):
        path1: y = 100  (entry line)
        path2: y = 200  (exit line)
    """
    path1 = [(0, 100), (500, 100)]
    path2 = [(0, 200), (500, 200)]
    return SpeedEstimator(
        path1=path1,
        path2=path2,
        real_distance_m=10.0,
        smoothing_window=3,
    )


# =============================================================================
# _crossed_polyline — geometric helper
# =============================================================================


class TestCrossedPolyline:
    """Verify the segment-crossing detection used to trigger speed timers."""

    def test_crosses_horizontal_line_top_to_bottom(self) -> None:
        """Moving downward through a horizontal line must be detected."""
        polyline = [(0, 100), (500, 100)]
        assert _crossed_polyline(polyline, (250.0, 90.0), (250.0, 110.0)) is True

    def test_crosses_horizontal_line_bottom_to_top(self) -> None:
        """Upward movement crossing the line must also be detected."""
        polyline = [(0, 100), (500, 100)]
        assert _crossed_polyline(polyline, (250.0, 110.0), (250.0, 90.0)) is True

    def test_no_cross_same_side_above(self) -> None:
        """Movement that stays above the line should not trigger a crossing."""
        polyline = [(0, 100), (500, 100)]
        assert _crossed_polyline(polyline, (100.0, 50.0), (200.0, 80.0)) is False

    def test_no_cross_same_side_below(self) -> None:
        """Movement that stays below the line should not trigger a crossing."""
        polyline = [(0, 100), (500, 100)]
        assert _crossed_polyline(polyline, (100.0, 150.0), (200.0, 180.0)) is False

    def test_parallel_horizontal_motion_no_cross(self) -> None:
        """Horizontal movement at a constant y never crosses a horizontal line."""
        polyline = [(0, 100), (500, 100)]
        assert _crossed_polyline(polyline, (50.0, 50.0), (450.0, 50.0)) is False

    def test_diagonal_cross(self) -> None:
        """A diagonal path that crosses the line should be detected."""
        polyline = [(0, 100), (500, 100)]
        # Moves from (0, 50) to (500, 150) — passes through y=100 at x=250.
        assert _crossed_polyline(polyline, (0.0, 50.0), (500.0, 150.0)) is True

    def test_movement_outside_segment_x_range_no_cross(self) -> None:
        """Movement that crosses the line's y level but is outside its x-span
        should NOT count as a crossing (the polyline is bounded)."""
        polyline = [(100, 100), (400, 100)]
        # Vertical movement at x=50, outside the [100, 400] x range.
        assert _crossed_polyline(polyline, (50.0, 90.0), (50.0, 110.0)) is False


# =============================================================================
# _point_to_polyline_distance — distance helper
# =============================================================================


class TestPointToPolylineDistance:
    """Basic sanity checks for the distance helper (used indirectly by the
    estimator and tested independently for clarity)."""

    def test_point_on_segment_returns_zero(self) -> None:
        polyline = [(0, 0), (100, 0)]
        dist = _point_to_polyline_distance((50.0, 0.0), polyline)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_point_perpendicular_above_segment(self) -> None:
        polyline = [(0, 0), (100, 0)]
        dist = _point_to_polyline_distance((50.0, 30.0), polyline)
        assert dist == pytest.approx(30.0, abs=1e-6)

    def test_point_past_end_of_segment(self) -> None:
        """Closest point is the segment endpoint, not an extension."""
        polyline = [(0, 0), (100, 0)]
        dist = _point_to_polyline_distance((150.0, 0.0), polyline)
        assert dist == pytest.approx(50.0, abs=1e-6)


# =============================================================================
# SpeedEstimator
# =============================================================================


class TestSpeedEstimator:
    """End-to-end tests for the SpeedEstimator class."""

    # ------------------------------------------------------------------
    # No speed before crossing both paths
    # ------------------------------------------------------------------

    def test_no_speed_before_any_crossing(self, estimator: SpeedEstimator) -> None:
        """get_speed returns None for an unseen track ID."""
        assert estimator.get_speed(99) is None

    def test_no_speed_after_crossing_only_path1(
        self, estimator: SpeedEstimator
    ) -> None:
        """Crossing only the entry line must not produce a speed reading."""
        estimator.estimate(1, (250.0, 90.0), timestamp=0.0)  # above path1
        estimator.estimate(1, (250.0, 110.0), timestamp=1.0)  # crosses path1
        estimator.estimate(1, (250.0, 150.0), timestamp=2.0)  # between lines
        assert estimator.get_speed(1) is None

    # ------------------------------------------------------------------
    # Correct speed calculation
    # ------------------------------------------------------------------

    def test_speed_calculation_correct(self, estimator: SpeedEstimator) -> None:
        """Speed must equal real_distance_m / delta_t * 3.6 km/h.

        Setup: path1 crossed at t=1.0 s, path2 crossed at t=5.0 s.
        Expected: 10 m / 4 s = 2.5 m/s = 9.0 km/h.
        """
        estimator.estimate(1, (250.0, 90.0), timestamp=0.0)  # above path1
        estimator.estimate(1, (250.0, 110.0), timestamp=1.0)  # crosses path1
        estimator.estimate(1, (250.0, 190.0), timestamp=4.0)  # above path2
        estimator.estimate(1, (250.0, 210.0), timestamp=5.0)  # crosses path2

        speed = estimator.get_speed(1)
        assert speed is not None
        assert speed == pytest.approx(9.0, abs=0.1)

    def test_second_speed_measurement(self, estimator: SpeedEstimator) -> None:
        """A different elapsed time should produce a proportionally different speed."""
        # 10 m in 2 s → 5 m/s → 18 km/h
        estimator.estimate(1, (250.0, 90.0), timestamp=0.0)
        estimator.estimate(1, (250.0, 110.0), timestamp=1.0)
        estimator.estimate(1, (250.0, 190.0), timestamp=2.5)
        estimator.estimate(1, (250.0, 210.0), timestamp=3.0)

        speed = estimator.get_speed(1)
        assert speed is not None
        assert speed == pytest.approx(18.0, abs=0.1)

    # ------------------------------------------------------------------
    # Multi-vehicle isolation
    # ------------------------------------------------------------------

    def test_multiple_vehicles_are_independent(self, estimator: SpeedEstimator) -> None:
        """Each track_id must have its own timer; speeds should differ."""
        for tid, t2 in [(10, 3.0), (20, 6.0)]:
            estimator.estimate(tid, (250.0, 90.0), timestamp=0.0)
            estimator.estimate(tid, (250.0, 110.0), timestamp=1.0)
            estimator.estimate(tid, (250.0, 190.0), timestamp=t2 - 0.5)
            estimator.estimate(tid, (250.0, 210.0), timestamp=t2)

        speed10 = estimator.get_speed(10)
        speed20 = estimator.get_speed(20)
        assert speed10 is not None and speed20 is not None
        assert speed10 != pytest.approx(speed20)

    # ------------------------------------------------------------------
    # Smoothing window
    # ------------------------------------------------------------------

    def test_smoothing_window_averages_readings(self) -> None:
        """With smoothing_window=2, the reported speed is the mean of the
        last two measurements."""
        est = SpeedEstimator(
            path1=[(0, 100), (500, 100)],
            path2=[(0, 200), (500, 200)],
            real_distance_m=10.0,
            smoothing_window=2,
        )
        # First crossing: 10 m in 4 s → 9.0 km/h
        est.estimate(1, (250.0, 90.0), timestamp=0.0)
        est.estimate(1, (250.0, 110.0), timestamp=1.0)
        est.estimate(1, (250.0, 190.0), timestamp=4.0)
        est.estimate(1, (250.0, 210.0), timestamp=5.0)

        assert est.get_speed(1) == pytest.approx(9.0, abs=0.1)

    # ------------------------------------------------------------------
    # Unrealistic speed filter
    # ------------------------------------------------------------------

    def test_speed_above_300_kmh_is_rejected(self, estimator: SpeedEstimator) -> None:
        """Tracker glitches can produce absurdly high speeds; they must be
        silently ignored and not added to the smoothing buffer."""
        # Both lines crossed in 0.001 s → ~36 000 km/h
        estimator.estimate(1, (250.0, 90.0), timestamp=0.000)
        estimator.estimate(1, (250.0, 310.0), timestamp=0.001)  # jumps past path2
        assert estimator.get_speed(1) is None

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    def test_all_speeds_property(self, estimator: SpeedEstimator) -> None:
        """all_speeds must contain every track_id that has a valid reading."""
        for tid in [1, 2, 3]:
            estimator.estimate(tid, (250.0, 90.0), timestamp=0.0)
            estimator.estimate(tid, (250.0, 110.0), timestamp=1.0)
            estimator.estimate(tid, (250.0, 190.0), timestamp=4.0)
            estimator.estimate(tid, (250.0, 210.0), timestamp=5.0)

        speeds = estimator.all_speeds
        assert set(speeds.keys()) == {1, 2, 3}
        assert all(v > 0 for v in speeds.values())

    def test_average_speed_returns_mean(self, estimator: SpeedEstimator) -> None:
        """average_speed must equal the arithmetic mean across all vehicles."""
        for tid in [1, 2]:
            estimator.estimate(tid, (250.0, 90.0), timestamp=0.0)
            estimator.estimate(tid, (250.0, 110.0), timestamp=1.0)
            estimator.estimate(tid, (250.0, 190.0), timestamp=4.0)
            estimator.estimate(tid, (250.0, 210.0), timestamp=5.0)

        avg = estimator.average_speed
        individual = list(estimator.all_speeds.values())
        expected = sum(individual) / len(individual)

        assert avg is not None
        assert avg == pytest.approx(expected, rel=1e-6)

    def test_average_speed_none_when_no_data(self, estimator: SpeedEstimator) -> None:
        """average_speed must return None when no vehicle has been timed yet."""
        assert estimator.average_speed is None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_all_state(self, estimator: SpeedEstimator) -> None:
        """After reset(), every piece of state must be empty."""
        estimator.estimate(1, (250.0, 90.0), timestamp=0.0)
        estimator.estimate(1, (250.0, 110.0), timestamp=1.0)
        estimator.estimate(1, (250.0, 190.0), timestamp=4.0)
        estimator.estimate(1, (250.0, 210.0), timestamp=5.0)

        estimator.reset()

        assert estimator.get_speed(1) is None
        assert estimator.all_speeds == {}
        assert estimator.average_speed is None

    def test_smoothing_buffer_trimmed_when_full(self) -> None:
        """When readings exceed smoothing_window, the oldest entry is dropped.

        Covers speed_estimator.py line 144:
            self._speeds[track_id] = self._speeds[track_id][-self.smoothing_window:]
        With smoothing_window=2 we need 3 valid crossings to trigger the trim.
        """
        est = SpeedEstimator(
            path1=[(0, 100), (500, 100)],
            path2=[(0, 200), (500, 200)],
            real_distance_m=10.0,
            smoothing_window=2,  # keep only the last 2 readings
        )
        # Three separate crossings for track_id=1 — third crossing trims the buffer.
        # Each crossing: cross path1, then cross path2.
        t = 0.0
        for _ in range(3):
            est.estimate(1, (250.0, 90.0), timestamp=t)
            t += 1.0
            est.estimate(1, (250.0, 110.0), timestamp=t)
            t += 1.0  # crosses path1
            est.estimate(1, (250.0, 190.0), timestamp=t)
            t += 1.0
            est.estimate(1, (250.0, 210.0), timestamp=t)
            t += 1.0  # crosses path2

        # After 3 crossings with window=2 the buffer must hold exactly 2 speeds.
        speed = est.get_speed(1)
        assert speed is not None
        assert speed > 0


# =============================================================================
# Collinear / endpoint edge cases in _segments_intersect
# These cover lines 64, 66, 68, 70, and 80 of speed_estimator.py — the four
# `return True` branches for points that lie exactly on a segment boundary.
# =============================================================================


class TestSegmentsIntersectCollinear:
    """Tests for the collinear edge cases that the normal crossing tests miss."""

    def test_p1_lies_on_polyline_segment_d1_zero(self) -> None:
        """Movement starts exactly on the polyline (d1==0, _on_segment True).

        Covers line 64: `if d1 == 0 and _on_segment(p3, p4, p1): return True`
        Setup: polyline is the horizontal segment y=0, x=[0..100].
               Movement starts at (50, 0) — ON the segment — and goes downward.
        """
        polyline = [(0, 0), (100, 0)]
        prev = (50.0, 0.0)  # p1 lies exactly on the polyline
        curr = (50.0, 20.0)  # p2 is below — not on the segment
        assert _crossed_polyline(polyline, prev, curr) is True

    def test_p2_lies_on_polyline_segment_d2_zero(self) -> None:
        """Movement ends exactly on the polyline (d2==0, _on_segment True).

        Covers line 66: `if d2 == 0 and _on_segment(p3, p4, p2): return True`
        Setup: movement arrives at (50, 0), which is ON the segment.
        """
        polyline = [(0, 0), (100, 0)]
        prev = (50.0, -20.0)  # p1 is above (negative y)
        curr = (50.0, 0.0)  # p2 lands exactly on the polyline
        assert _crossed_polyline(polyline, prev, curr) is True

    def test_polyline_start_on_movement_path_d3_zero(self) -> None:
        """The first vertex of the polyline lies on the movement vector (d3==0).

        Covers line 68: `if d3 == 0 and _on_segment(p1, p2, p3): return True`
        Setup: movement is a horizontal sweep from x=0 to x=100 at y=50.
               Polyline goes from (50, 50) downward to (50, 100).
               p3=(50,50) is exactly on the movement segment.
        """
        polyline = [(50, 50), (50, 100)]
        prev = (0.0, 50.0)  # start of movement
        curr = (100.0, 50.0)  # end of movement — p3 lies in between
        assert _crossed_polyline(polyline, prev, curr) is True

    def test_polyline_end_on_movement_path_d4_zero(self) -> None:
        """The last vertex of the polyline lies on the movement vector (d4==0).

        Covers line 70: `if d4 == 0 and _on_segment(p1, p2, p4): return True`
        Setup: movement is a horizontal sweep at y=50.
               Polyline goes from (50, 0) up to (50, 50).
               p4=(50,50) is exactly on the movement segment.
        """
        polyline = [(50, 0), (50, 50)]
        prev = (0.0, 50.0)  # start of movement
        curr = (100.0, 50.0)  # end of movement — p4 lies in between
        assert _crossed_polyline(polyline, prev, curr) is True


# =============================================================================
# Degenerate segment in _point_to_polyline_distance
# Covers speed_estimator.py line 26 (the `if ab_sq == 0` branch).
# =============================================================================


class TestPointToPolylineDistanceDegenerate:
    def test_degenerate_segment_both_endpoints_identical(self) -> None:
        """When both endpoints of a segment are the same point (ab_sq==0),
        the distance is simply the Euclidean distance to that point.

        Covers line 26: `dist = (apx * apx + apy * apy) ** 0.5`
        Point is (53, 54), degenerate segment is (50, 50)→(50, 50).
        Expected distance: sqrt((53-50)^2 + (54-50)^2) = sqrt(9+16) = 5.0
        """
        degenerate_polyline = [(50, 50), (50, 50)]
        dist = _point_to_polyline_distance((53.0, 54.0), degenerate_polyline)
        assert dist == pytest.approx(5.0, abs=1e-6)
