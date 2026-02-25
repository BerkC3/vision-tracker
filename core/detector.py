from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device (shared by detector & tracker)."""
    if device == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
            return "cuda:0"
        logger.warning("CUDA not available! Running on CPU - expect slow performance.")
        return "cpu"
    return device


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "auto",
        classes: list[int] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes or list(COCO_VEHICLE_NAMES.keys())
        logger.info(f"Detector ready on {self.device} | model={model_path}")

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
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].cpu().numpy(),
                        class_id=cls_id,
                        class_name=COCO_VEHICLE_NAMES.get(cls_id, "unknown"),
                        confidence=float(box.conf[0]),
                    )
                )
        return detections

    def detect_raw(self, frame: np.ndarray) -> list:
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
