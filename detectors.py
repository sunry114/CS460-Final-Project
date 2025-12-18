from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import cv2
import torch
from ultralytics import YOLO
import torchvision
from torchvision.transforms import functional as F


@dataclass
class ModelSpec:
    name: str
    kind: str  # "yolo" or "torchvision"
    yolo_weight: Optional[str] = None
    min_score_for_ap: float = 0.001


class YOLODetector:
    def __init__(self, weight: str, device: str):
        self.model = YOLO(weight)
        self.device = device
        names = self.model.names
        self.class_id = None
        for k, v in names.items():
            if str(v).strip().lower() == "traffic light":
                self.class_id = int(k)
                break
        if self.class_id is None:
            raise RuntimeError(f"Cannot find 'traffic light' in YOLO names: {names}")

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        r = self.model.predict(
            source=img_bgr,
            verbose=False,
            device=self.device,
            imgsz=640,
            conf=0.001,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(np.int32)

        keep = cls == self.class_id
        boxes = boxes[keep]
        conf = conf[keep]
        if boxes.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)

        return np.concatenate([boxes, conf[:, None]], axis=1)


class FasterRCNNDetector:
    """Torchvision Faster R-CNN COCO traffic light label id = 10."""
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device).eval()
        self.tl_label = 10

    @torch.inference_mode()
    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = F.to_tensor(img_rgb).to(self.device)
        out = self.model([t])[0]

        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"].detach().cpu().numpy().astype(np.int32)

        keep = labels == self.tl_label
        boxes = boxes[keep]
        scores = scores[keep]
        if boxes.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)

        return np.concatenate([boxes, scores[:, None]], axis=1)


def build_detectors(models, device: str) -> Dict[str, object]:
    detectors = {}
    for m in models:
        if m.kind == "yolo":
            detectors[m.name] = YOLODetector(m.yolo_weight, device=device)
        else:
            detectors[m.name] = FasterRCNNDetector(device=device)
    return detectors
