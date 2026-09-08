#!/usr/bin/env python3
"""
EBiM Task 3 - 视觉检测模块
从摄像头读取图像，使用 YOLOv8 检测物体，输出 3D 位置。
与 task3_autonomous.py 中的 vision_callback.py 功能一致，但支持实机模式。

支持的摄像头:
  - head: 头部全局摄像头 (ZED)
  - wrist_left: 左手腕摄像头
  - wrist_right: 右手腕摄像头
"""

import os
import time
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

# YOLO model (loaded lazily)
_yolo_model = None
_model_lock = threading.Lock()

# Single source of truth for the checkpoint label order.  It must match
# vision_callback.OBJECT_CLASSES and data/task3_yolo/task3.yaml exactly.
TASK3_CLASSES = [
    "plate", "cup", "bowl", "spoon", "head",
    "sink", "recycling_bin", "seat_area", "simple_tray", "bean",
]


def model_class_names(model_names) -> set[str]:
    """Normalize Ultralytics list/dict class metadata without loading a model."""
    values = model_names.values() if isinstance(model_names, dict) else model_names
    return {str(value) for value in values}


def missing_task3_classes(model_names) -> list[str]:
    """Return expected task classes absent from a checkpoint metadata table."""
    available = model_class_names(model_names)
    return [name for name in TASK3_CLASSES if name not in available]


def load_yolo_model(model_path: str = None):
    """Load YOLOv8 model (thread-safe)."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    with _model_lock:
        if _yolo_model is not None:
            return _yolo_model

        from ultralytics import YOLO

        if model_path is None:
            # Never reinterpret a generic COCO checkpoint as Task 3 labels.
            default_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/ebim_task3.pt"),
                "models/ebim_task3.pt",
            ]
            for p in default_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        if model_path is None:
            raise FileNotFoundError("No YOLO model found. Train one with train_yolo.py")

        print(f"Loading YOLO model from: {model_path}")
        candidate = YOLO(model_path)
        missing = missing_task3_classes(candidate.names)
        if missing:
            raise ValueError(
                "checkpoint is not an EBiM Task 3 model; missing classes: "
                + ", ".join(missing))
        _yolo_model = candidate
        return _yolo_model


def detect_objects(image: np.ndarray, model_path: str = None,
                   conf_threshold: float = 0.35,
                   classes: Optional[List[int]] = None) -> List[Dict]:
    """
    Detect objects in an image using YOLOv8.

    Args:
        image: BGR image numpy array (H, W, 3)
        model_path: Path to YOLO model weights
        conf_threshold: Confidence threshold
        classes: List of class indices to detect (None = all)

    Returns:
        List of detections, each with:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: [x1, y1, x2, y2] (pixel coordinates)
            - center: [cx, cy] (pixel coordinates)
    """
    model = load_yolo_model(model_path)

    results = model(image, conf=conf_threshold, classes=classes, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = result.boxes.conf.cpu().numpy()  # (N,)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        names = getattr(result, "names", {})
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            class_name = str(names.get(cls_id, "")) if isinstance(names, dict) else (
                str(names[cls_id]) if 0 <= cls_id < len(names) else "")
            if class_name not in TASK3_CLASSES:
                continue

            detections.append({
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(cx), float(cy)],
            })

    return detections


def get_object_3d_position(detection: Dict, depth_image: np.ndarray,
                           camera_info: Dict) -> Optional[np.ndarray]:
    """
    Estimate 3D position of detected object using depth image.

    Args:
        detection: Detection dict from detect_objects()
        depth_image: Depth image (H, W) in meters
        camera_info: Camera intrinsics dict with 'fx', 'fy', 'cx', 'cy'

    Returns:
        [x, y, z] position in camera frame (meters), or None if invalid
    """
    cx, cy = detection["center"]
    h, w = depth_image.shape[:2]

    # Sample depth around center (5x5 window)
    x1 = max(0, int(cx - 2))
    x2 = min(w - 1, int(cx + 2))
    y1 = max(0, int(cy - 2))
    y2 = min(h - 1, int(cy + 2))

    depth_patch = depth_image[y1:y2+1, x1:x2+1]
    valid_depths = depth_patch[(depth_patch > 0.01) & (depth_patch < 10.0)]

    if len(valid_depths) < 3:
        return None

    z = np.median(valid_depths)

    fx = camera_info.get("fx", 525.0)
    fy = camera_info.get("fy", 525.0)
    cam_cx = camera_info.get("cx", w / 2.0)
    cam_cy = camera_info.get("cy", h / 2.0)

    x = (cx - cam_cx) * z / fx
    y = (cy - cam_cy) * z / fy

    return np.array([x, y, z])


def select_best_detection(detections: List[Dict], target_class: str,
                          image_center: Tuple[float, float] = None) -> Optional[Dict]:
    """
    Select the best detection of a target class.
    Priority: highest confidence, then closest to image center.
    """
    candidates = [d for d in detections if d["class_name"] == target_class]
    if not candidates:
        return None

    if image_center is None:
        # Just return highest confidence
        return max(candidates, key=lambda d: d["confidence"])

    # Weighted score: confidence + proximity to center
    cx, cy = image_center
    best = None
    best_score = -1

    for d in candidates:
        dx = d["center"][0] - cx
        dy = d["center"][1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        max_dist = np.sqrt(cx * cx + cy * cy)
        center_score = 1.0 - dist / max_dist  # 0 to 1
        score = d["confidence"] * 0.7 + center_score * 0.3

        if score > best_score:
            best_score = score
            best = d

    return best


if __name__ == "__main__":
    # Quick test
    print("YOLO detection module loaded")
    print(f"Task 3 classes: {TASK3_CLASSES}")
    print(f"Number of classes: {len(TASK3_CLASSES)}")
