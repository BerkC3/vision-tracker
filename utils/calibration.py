from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _enable_dpi_awareness() -> None:
    """Fix mouse coordinate mismatch on Windows with display scaling."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            pass


def _fit_display(frame: np.ndarray, max_w: int = 1400, max_h: int = 800) -> tuple[np.ndarray, float]:
    """Resize frame to fit display, return (resized_frame, scale_factor)."""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame.copy(), 1.0
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


class PathCalibrator:
    """Interactive calibration for perspective views.

    Left-click: add points to path 1 (green)
    Right-click: add points to path 2 (orange)
    'c': close/finish current paths
    'r': reset everything
    'q': skip calibration, use defaults
    """

    def __init__(self) -> None:
        self._path1: list[tuple[int, int]] = []
        self._path2: list[tuple[int, int]] = []
        self._frame: np.ndarray | None = None
        self._display: np.ndarray | None = None
        self._scale: float = 1.0
        self._done = False

    def _to_orig(self, x: int, y: int) -> tuple[int, int]:
        return int(x / self._scale), int(y / self._scale)

    def _to_disp(self, pt: tuple[int, int]) -> tuple[int, int]:
        return int(pt[0] * self._scale), int(pt[1] * self._scale)

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self._done:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._path1.append(self._to_orig(x, y))
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._path2.append(self._to_orig(x, y))
            self._redraw()

    def _redraw(self) -> None:
        self._display, _ = _fit_display(self._frame, 1400, 800)
        self._draw_instructions(self._display)

        # Path 1 (green)
        if self._path1:
            pts = np.array([self._to_disp(p) for p in self._path1], dtype=np.int32)
            cv2.polylines(self._display, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
            for i, p in enumerate(self._path1):
                dp = self._to_disp(p)
                cv2.circle(self._display, dp, 6, (0, 255, 0), -1)
                cv2.putText(self._display, str(i + 1), (dp[0] + 8, dp[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Path 2 (orange)
        if self._path2:
            pts = np.array([self._to_disp(p) for p in self._path2], dtype=np.int32)
            cv2.polylines(self._display, [pts], False, (0, 140, 255), 3, cv2.LINE_AA)
            for i, p in enumerate(self._path2):
                dp = self._to_disp(p)
                cv2.circle(self._display, dp, 6, (0, 140, 255), -1)
                cv2.putText(self._display, str(i + 1), (dp[0] + 8, dp[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

    def _draw_instructions(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        scale = max(w / 1920, 0.5)
        pad = int(12 * scale)
        font_title = 0.7 * scale
        font_body = 0.5 * scale
        font_hint = 0.45 * scale
        thick_title = max(1, int(2 * scale))
        thick_body = max(1, int(1 * scale))
        line_h = int(24 * scale)
        panel_w = int(550 * scale)
        panel_h = int(110 * scale)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        y0 = pad + line_h
        cv2.putText(img, "SPEED PATH CALIBRATION", (pad, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, font_title, (0, 255, 255), thick_title)
        cv2.putText(img, "Left-click: draw PATH 1 (start zone) - green", (pad, y0 + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_body, (0, 255, 0), thick_body)
        cv2.putText(img, "Right-click: draw PATH 2 (end zone) - orange", (pad, y0 + line_h * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_body, (0, 140, 255), thick_body)
        cv2.putText(img, "'c' = confirm  |  'r' = reset  |  'q' = skip (use defaults)", (pad, y0 + line_h * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_hint, (200, 200, 200), thick_body)

    def calibrate(self, frame: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
        _enable_dpi_awareness()
        self._frame = frame.copy()
        self._display, self._scale = _fit_display(frame, 1400, 800)
        self._draw_instructions(self._display)
        self._path1.clear()
        self._path2.clear()
        self._done = False

        win = "Vision-Track | Speed Path Calibration"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            cv2.imshow(win, self._display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("r"):
                self._path1.clear()
                self._path2.clear()
                self._redraw()

            elif key == ord("c"):
                if len(self._path1) >= 2 and len(self._path2) >= 2:
                    self._done = True
                    break
                else:
                    h = self._display.shape[0]
                    cv2.putText(self._display, "Need at least 2 points per path!",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            elif key == ord("q"):
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        logger.info(f"Path 1: {len(self._path1)} points | Path 2: {len(self._path2)} points")
        return list(self._path1), list(self._path2)


ROI_COLORS = [
    (255, 100, 0),   # blue-orange
    (0, 100, 255),   # red-orange
    (100, 255, 0),   # green-ish
    (255, 0, 200),   # pink
    (0, 255, 200),   # cyan
]


class ROICalibrator:
    """Interactive multi-ROI calibrator.

    Left-click: add polygon points
    Right-click: finish current ROI (min 3 points)
    'n': start new ROI
    'r': reset current ROI
    'c': confirm all ROIs and exit
    'q': skip (no ROIs)
    """

    def __init__(self) -> None:
        self._current_points: list[tuple[int, int]] = []
        self._completed: list[list[tuple[int, int]]] = []
        self._frame: np.ndarray | None = None
        self._display: np.ndarray | None = None
        self._scale: float = 1.0
        self._current_done = False

    def _to_orig(self, x: int, y: int) -> tuple[int, int]:
        return int(x / self._scale), int(y / self._scale)

    def _to_disp(self, pt: tuple[int, int]) -> tuple[int, int]:
        return int(pt[0] * self._scale), int(pt[1] * self._scale)

    def _color(self, index: int) -> tuple[int, int, int]:
        return ROI_COLORS[index % len(ROI_COLORS)]

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self._current_done:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_points.append(self._to_orig(x, y))
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN and len(self._current_points) >= 3:
            self._current_done = True

    def _redraw(self) -> None:
        self._display, _ = _fit_display(self._frame, 1400, 800)
        self._draw_instructions(self._display)
        # Draw completed ROIs
        for i, poly in enumerate(self._completed):
            self._draw_polygon(self._display, poly, self._color(i), i + 1)
        # Draw current ROI in progress
        if self._current_points:
            ci = len(self._completed)
            self._draw_polygon(self._display, self._current_points, self._color(ci), ci + 1, closed=False)

    def _draw_instructions(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        scale = max(w / 1920, 0.5)
        pad = int(12 * scale)
        font_title = 0.7 * scale
        font_body = 0.5 * scale
        thick_title = max(1, int(2 * scale))
        thick_body = max(1, int(1 * scale))
        line_h = int(24 * scale)
        panel_w = int(620 * scale)
        panel_h = int(120 * scale)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        count = len(self._completed)
        title = f"ROI SETUP - {count} zone(s) defined" if count else "ROI SETUP (Restricted Zones)"
        y0 = pad + line_h
        cv2.putText(img, title, (pad, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, font_title, (0, 255, 255), thick_title)
        cv2.putText(img, "Left-click: add points  |  Right-click: finish polygon", (pad, y0 + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_body, (255, 100, 0), thick_body)
        cv2.putText(img, "'n' = new ROI  |  'r' = reset current  |  'c' = confirm all", (pad, y0 + line_h * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_body, (200, 200, 200), thick_body)
        cv2.putText(img, "'q' = skip (no ROI)", (pad, y0 + line_h * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_body, (150, 150, 150), thick_body)

    def _draw_polygon(self, img: np.ndarray, points: list[tuple[int, int]],
                       color: tuple[int, int, int], label_num: int, closed: bool = True) -> None:
        disp_pts = [self._to_disp(p) for p in points]
        for i, dp in enumerate(disp_pts):
            cv2.circle(img, dp, 5, color, -1)
            if i > 0:
                cv2.line(img, disp_pts[i - 1], dp, color, 2)
        if len(disp_pts) >= 3:
            pts_arr = np.array(disp_pts, dtype=np.int32)
            if closed:
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts_arr], color)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [pts_arr], closed, color, 2)
            cv2.putText(img, f"ROI #{label_num}", (disp_pts[0][0] + 8, disp_pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calibrate(self, frame: np.ndarray) -> list[list[list[int]]] | None:
        """Returns list of polygons: [[[x,y], [x,y], ...], [...], ...]"""
        _enable_dpi_awareness()
        self._frame = frame.copy()
        self._display, self._scale = _fit_display(frame, 1400, 800)
        self._current_points.clear()
        self._completed.clear()
        self._current_done = False

        win = "Vision-Track | ROI Calibration"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._mouse_cb)
        self._draw_instructions(self._display)

        while True:
            cv2.imshow(win, self._display)
            key = cv2.waitKey(30) & 0xFF

            if self._current_done:
                # Current ROI finished - save it
                self._completed.append(list(self._current_points))
                self._current_points.clear()
                self._current_done = False
                self._redraw()

            if key == ord("n"):
                # Start new ROI (save current if valid)
                if len(self._current_points) >= 3:
                    self._completed.append(list(self._current_points))
                self._current_points.clear()
                self._current_done = False
                self._redraw()

            elif key == ord("r"):
                self._current_points.clear()
                self._current_done = False
                self._redraw()

            elif key == ord("c"):
                if len(self._current_points) >= 3:
                    self._completed.append(list(self._current_points))
                if self._completed:
                    break
                # Flash warning
                h = self._display.shape[0]
                cv2.putText(self._display, "Draw at least one ROI (3+ points)!",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif key == ord("q"):
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        result = [[[p[0], p[1]] for p in poly] for poly in self._completed]
        logger.info(f"ROI calibration: {len(result)} zone(s)")
        return result
