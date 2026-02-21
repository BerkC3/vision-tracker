from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO

from .detector import COCO_VEHICLE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    center: tuple[float, float]
    trail: list[tuple[float, float]] = field(default_factory=list)


class VehicleTracker:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "auto",
        classes: list[int] | None = None,
        tracker_config: str = "botsort.yaml",
        trail_length: int = 30,
        imgsz: int = 1920,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.classes = classes or list(COCO_VEHICLE_NAMES.keys())
        self.tracker_config = tracker_config
        self.trail_length = trail_length
        self._trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._seen_ids: set[int] = set()
        logger.info(f"Tracker ready on {self.device} | imgsz={imgsz} | tracker={tracker_config}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
                return "cuda:0"
            logger.warning("CUDA not available! Running on CPU - expect slow performance.")
            logger.warning("Install CUDA torch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return "cpu"
        return device

    def update(self, frame: np.ndarray) -> list[TrackedVehicle]:
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            persist=True,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
        )

        vehicles: list[TrackedVehicle] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, track_id in zip(r.boxes, r.boxes.id):
                tid = int(track_id)
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cx = float((xyxy[0] + xyxy[2]) / 2)
                cy = float((xyxy[1] + xyxy[3]) / 2)

                self._trails[tid].append((cx, cy))
                if len(self._trails[tid]) > self.trail_length:
                    self._trails[tid] = self._trails[tid][-self.trail_length:]

                self._seen_ids.add(tid)

                vehicles.append(TrackedVehicle(
                    track_id=tid,
                    bbox=xyxy,
                    class_id=cls_id,
                    class_name=COCO_VEHICLE_NAMES.get(cls_id, "unknown"),
                    confidence=float(box.conf[0]),
                    center=(cx, cy),
                    trail=list(self._trails[tid]),
                ))
        return vehicles

    @property
    def total_unique_vehicles(self) -> int:
        return len(self._seen_ids)

    def reset(self) -> None:
        self._trails.clear()
        self._seen_ids.clear()
