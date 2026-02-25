from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: str | int = 0) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._resolved_path: str | int = source

    def open(self) -> None:
        source = self._source

        if isinstance(source, str) and (
            "youtube.com" in source or "youtu.be" in source
        ):
            source = self._resolve_youtube(source)

        self._resolved_path = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._source}")

        logger.info(
            f"Video opened: {self.width}x{self.height} @ {self.fps:.1f} FPS "
            f"({self.frame_count} frames)"
        )

    @staticmethod
    def _resolve_youtube(url: str) -> str:
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "best[height<=1080]", "-g", url],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                stream_url = result.stdout.strip()
                logger.info("YouTube stream URL resolved")
                return stream_url
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self._cap is None:
            self.open()
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def seek_start(self) -> None:
        """Seek back to the first frame. No-op for live/stream sources."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
        logger.info(f"Video writer: {output_path} ({width}x{height} @ {fps:.1f})")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
