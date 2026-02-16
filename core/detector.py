from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    bbox: np.ndarray       # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "auto",
        classes: list[int] | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes or list(COCO_VEHICLE_NAMES.keys())
        logger.info(f"Detector ready on {self.device} | model={model_path}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA device: {name}")
                return "cuda:0"
            return "cpu"
        return device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(Detection(
                    bbox=box.xyxy[0].cpu().numpy(),
                    class_id=cls_id,
                    class_name=COCO_VEHICLE_NAMES.get(cls_id, "unknown"),
                    confidence=float(box.conf[0]),
                ))
        return detections

    def detect_raw(self, frame: np.ndarray):
        """Return raw ultralytics Results for tracker integration."""
        return self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )
