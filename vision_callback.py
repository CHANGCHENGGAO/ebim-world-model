#!/usr/bin/env python3
from __future__ import annotations
"""
Vision System for EBiM Task 3 — Phase II

- Subscribes to configured RGB-D cameras (ROS2 Image + CameraInfo) when available
- Runs YOLOv8 detection on each camera stream
- Publishes object poses with confidence and timestamp
- Low confidence → re-observe (NO fallback to hardcoded coordinates)
- Supports both sim (Isaac Sim camera API) and real (ROS2 subscriptions) modes

Object set: plate, cup, bowl, spoon, head, sink, recycling_bin, seat_area

Data Annotation Format (YOLO format: one .txt per image, one row per object):
  class_id  cx_normalized  cy_normalized  w_normalized  h_normalized

  class_id mapping:
    0: plate        — round flat dish, ~25cm diameter
    1: cup           — cylindrical drink container with handle
    2: bowl          — round deep container, ~15cm diameter
    3: spoon         — long-handled utensil with oval bowl end
    4: head          — human head/face region (mannequin or person)
    5: sink          — rectangular basin fixture
    6: recycling_bin — tall rectangular container (Ikea knock box)
    7: seat_area     — chair/stool surface region
    8: simple_tray   — flat rectangular serving tray
    9: bean          — small round dark coffee bean (~1cm)

  Bounding box values are normalized to [0,1] relative to image W/H.
  Example label row: 2 0.512 0.384 0.150 0.120  (bowl centered slightly left)
"""

import argparse
import hashlib
import math
import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple

from b2_contract import ContractError, load_contract, quaternion_transform_point

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Image, CameraInfo
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration
    from rclpy.time import Time
    import tf2_ros
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False
    Node = object

# ============================================================
# Object definitions and annotation format
# ============================================================

# EBiM Task 3 target objects with class IDs for labeling
OBJECT_CLASSES = {
    0: "plate",
    1: "cup",
    2: "bowl",
    3: "spoon",
    4: "head",
    5: "sink",
    6: "recycling_bin",
    7: "seat_area",
    8: "simple_tray",
    9: "bean",
}

# Reverse mapping
NAME_TO_CLASS = {v: k for k, v in OBJECT_CLASSES.items()}

# Annotation format reference (also used for dataset.yaml generation)
DATASET_YAML = """\
# EBiM Task 3 Object Detection Dataset
path: ./dataset
train: images/train
val: images/val
names:
  0: plate
  1: cup
  2: bowl
  3: spoon
  4: head
  5: sink
  6: recycling_bin
  7: seat_area
  8: simple_tray
  9: bean
"""

# Minimum confidence threshold per object type
MIN_CONFIDENCE = {
    "plate": 0.35,
    "cup": 0.35,
    "bowl": 0.40,
    "spoon": 0.30,
    "head": 0.50,
    "sink": 0.30,
    "recycling_bin": 0.35,
    "seat_area": 0.25,
    "simple_tray": 0.30,
    "bean": 0.20,
}


def missing_task3_classes(model_names) -> List[str]:
    """Return required task classes absent from a checkpoint's class table."""
    names = {str(name) for name in model_names}
    return sorted(set(OBJECT_CLASSES.values()) - names)


def lighting_quality(image: np.ndarray) -> Dict[str, float | bool]:
    """Measure exposure usability without rejecting a merely dim image."""
    if image is None or image.size == 0 or image.ndim < 2:
        return {"usable": False, "mean_luma": 0.0, "contrast": 0.0,
                "dark_fraction": 1.0, "bright_fraction": 0.0}
    rgb = image[:, :, :3].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    luma = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    mean_luma = float(luma.mean() / 255.0)
    contrast = float(luma.std() / 255.0)
    dark_fraction = float((luma <= 3.0).mean())
    bright_fraction = float((luma >= 252.0).mean())
    usable = contrast >= 0.025 and dark_fraction < 0.98 and bright_fraction < 0.98
    return {"usable": usable, "mean_luma": mean_luma, "contrast": contrast,
            "dark_fraction": dark_fraction, "bright_fraction": bright_fraction}


def adaptive_gamma(image: np.ndarray, target_median: float = 0.50) -> float:
    """Return a bounded gamma that moves image median toward target_median."""
    if image is None or image.size == 0:
        return 1.0
    values = image[:, :, :3] if image.ndim == 3 else image
    median = float(np.median(values) / 255.0)
    if not 0.01 < median < 0.99:
        return 1.0
    gamma = math.log(target_median) / math.log(median)
    return float(np.clip(gamma, 0.45, 2.20))

class VisionSystem:
    """ROS2-based vision system for EBiM Task 3.

    Subscribes to RGB-D camera topics, runs YOLOv8, publishes
    object positions with confidence and timestamp.
    Never falls back to hardcoded coordinates.
    """

    def __init__(self, node: Node, robot_mode: str = "sim",
                 config_path: Optional[str] = None, model_path: Optional[str] = None):
        self.node = node
        self.robot_mode = robot_mode
        self.contract = load_contract(config_path, robot_mode)
        if (not self.contract.get("confirmed", False) and
                os.environ.get("ALLOW_UNCONFIRMED_IO", "0") != "1"):
            raise ContractError(
                f"{robot_mode} camera contract is UNCONFIRMED; refusing perception output"
            )
        if not self.contract.get("vision_3d_available", False):
            raise ContractError(
                "The selected ROS2 contract does not provide depth and CameraInfo; "
                "internal RGB-D object-pose publication is unavailable. Use a separately "
                "validated 3D perception adapter, not inferred coordinates.")
        self.world_frame = self.contract["world_frame"]
        self.topics = self.contract["topics"]
        self.model_path = model_path or os.environ.get("VISION_MODEL_PATH", "")
        self.detection_max_age = float(os.environ.get("DETECTION_MAX_AGE", "3.0"))
        self.detection_period = float(os.environ.get("VISION_PERIOD_SECONDS", "0.5"))
        self.camera_sync_tolerance = float(os.environ.get("CAMERA_SYNC_TOLERANCE", "0.15"))
        self.model = None
        self._lock = threading.Lock()

        # Latest detections: {name: (x, y, z, conf, timestamp)}
        self._detections: Dict[str, Tuple[float, float, float, float, float]] = {}
        self._latest_beans: List[Tuple[float, float, float, float]] = []
        self._lighting_status: Dict[str, Dict[str, float | bool]] = {}

        # Camera data buffers
        self._head_rgb = None
        self._head_depth = None
        self._head_info = None
        self._left_rgb = None
        self._left_depth = None
        self._right_rgb = None
        self._right_depth = None
        self._camera_frames = {"head": "", "left_wrist": "", "right_wrist": ""}
        self._rgb_times = {"head": 0.0, "left_wrist": 0.0, "right_wrist": 0.0}
        self._depth_times = {"head": 0.0, "left_wrist": 0.0, "right_wrist": 0.0}
        self._intrinsics = {"head": None, "left_wrist": None, "right_wrist": None}

        self._head_rgb_topic = self.topics["head_rgb"]
        self._head_depth_topic = self.topics["head_depth"]
        self._head_info_topic = self.topics["head_camera_info"]
        self._left_rgb_topic = self.topics["left_wrist_rgb"]
        self._left_depth_topic = self.topics["left_wrist_depth"]
        self._left_info_topic = self.topics["left_wrist_camera_info"]
        self._right_rgb_topic = self.topics["right_wrist_rgb"]
        self._right_depth_topic = self.topics["right_wrist_depth"]
        self._right_info_topic = self.topics["right_wrist_camera_info"]

        # Publishers
        self._obj_pub = node.create_publisher(
            JointState, self.topics["vision_objects"], 10)
        self._bean_pub = node.create_publisher(
            JointState, self.topics["vision_beans"], 10)
        self._status_pub = node.create_publisher(
            String, "/vision/status", 10)

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        # Load YOLO model
        self._load_model()
        if self.model is None:
            raise RuntimeError(
                f"vision model unavailable at {self.model_path!r}; runtime downloads are disabled"
            )

        # Subscribe to camera topics
        self._subscribe_cameras()

        # Detection timer
        self._detect_timer = node.create_timer(self.detection_period, self._detect_callback)
        self._detect_count = 0

        node.get_logger().info(
            f"VisionSystem initialized — mode={robot_mode}, "
            f"model={'loaded' if self.model else 'NONE'}, "
            f"objects={list(OBJECT_CLASSES.values())}"
        )

    def _load_model(self):
        """Load a packaged Task 3 model. Runtime downloads are forbidden."""
        if not self.model_path or not os.path.isfile(self.model_path):
            self.node.get_logger().error(
                f"Vision checkpoint not found: {self.model_path!r}"
            )
            self.model = None
            return
        try:
            from ultralytics import YOLO
            candidate = YOLO(self.model_path)
            model_names = set(candidate.names.values())
            missing_names = missing_task3_classes(model_names)
            if missing_names:
                raise RuntimeError(
                    "checkpoint does not implement the Task 3 class contract; "
                    f"missing={missing_names}"
                )
            with open(self.model_path, "rb") as checkpoint_file:
                digest = hashlib.sha256(checkpoint_file.read()).hexdigest()
            self.model = candidate
            self.node.get_logger().info(
                f"YOLO Task 3 model loaded: path={self.model_path} sha256={digest} "
                f"classes={sorted(model_names)}"
            )
        except Exception as e:
            self.node.get_logger().error(f"YOLO load failed: {e}")
            self.model = None

    def _subscribe_cameras(self):
        """Subscribe to RGB-D camera topics."""
        if not _HAS_ROS:
            self.node.get_logger().warn("ROS2 not available, skipping camera subscriptions")
            return

        from sensor_msgs.msg import Image as ImageMsg, CameraInfo as CameraInfoMsg

        self._head_rgb_sub = self.node.create_subscription(
            ImageMsg, self._head_rgb_topic, self._head_rgb_cb, 10)
        self._head_depth_sub = self.node.create_subscription(
            ImageMsg, self._head_depth_topic, self._head_depth_cb, 10)
        self._head_info_sub = self.node.create_subscription(
            CameraInfoMsg, self._head_info_topic, self._head_info_cb, 10)

        self._left_rgb_sub = self.node.create_subscription(
            ImageMsg, self._left_rgb_topic, self._left_rgb_cb, 10)
        self._left_depth_sub = self.node.create_subscription(
            ImageMsg, self._left_depth_topic, self._left_depth_cb, 10)
        self._left_info_sub = self.node.create_subscription(
            CameraInfoMsg, self._left_info_topic,
            lambda msg: self._camera_info_cb(msg, "left_wrist"), 10)

        self._right_rgb_sub = self.node.create_subscription(
            ImageMsg, self._right_rgb_topic, self._right_rgb_cb, 10)
        self._right_depth_sub = self.node.create_subscription(
            ImageMsg, self._right_depth_topic, self._right_depth_cb, 10)
        self._right_info_sub = self.node.create_subscription(
            CameraInfoMsg, self._right_info_topic,
            lambda msg: self._camera_info_cb(msg, "right_wrist"), 10)

        self.node.get_logger().info(
            f"Subscribed to cameras: head={self._head_rgb_topic}, "
            f"left={self._left_rgb_topic}, right={self._right_rgb_topic}"
        )

    # --- Camera callbacks ---

    def _head_rgb_cb(self, msg):
        self._head_rgb = self._ros_image_to_numpy(msg)
        self._rgb_times["head"] = time.monotonic()
        self._camera_frames["head"] = msg.header.frame_id

    def _head_depth_cb(self, msg):
        self._head_depth = self._ros_image_to_numpy(msg)
        self._depth_times["head"] = time.monotonic()

    def _head_info_cb(self, msg):
        self._camera_info_cb(msg, "head")

    def _camera_info_cb(self, msg, camera_name):
        self._intrinsics[camera_name] = (
            float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5])
        )
        if not self._camera_frames[camera_name]:
            self._camera_frames[camera_name] = msg.header.frame_id

    def _left_rgb_cb(self, msg):
        self._left_rgb = self._ros_image_to_numpy(msg)
        self._rgb_times["left_wrist"] = time.monotonic()
        self._camera_frames["left_wrist"] = msg.header.frame_id

    def _left_depth_cb(self, msg):
        self._left_depth = self._ros_image_to_numpy(msg)
        self._depth_times["left_wrist"] = time.monotonic()

    def _right_rgb_cb(self, msg):
        self._right_rgb = self._ros_image_to_numpy(msg)
        self._rgb_times["right_wrist"] = time.monotonic()
        self._camera_frames["right_wrist"] = msg.header.frame_id

    def _right_depth_cb(self, msg):
        self._right_depth = self._ros_image_to_numpy(msg)
        self._depth_times["right_wrist"] = time.monotonic()

    @staticmethod
    def _ros_image_to_numpy(msg):
        """Convert sensor_msgs/Image to numpy array."""
        try:
            import numpy as np
            if msg.encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding in ("rgba8", "bgra8"):
                    arr = arr.reshape(msg.height, msg.width, 4)[:, :, :3]
                else:
                    arr = arr.reshape(msg.height, msg.width, 3)
                if msg.encoding in ("bgr8", "bgra8"):
                    arr = arr[:, :, ::-1].copy()
                return arr
            elif msg.encoding in ("32FC1", "16UC1"):
                dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
                arr = np.frombuffer(msg.data, dtype=dtype)
                return arr.reshape(msg.height, msg.width)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                return arr.reshape(msg.height, msg.width, -1)
        except Exception:
            return None

    # --- Detection ---

    def _detect_callback(self):
        """Run detection on synchronized head and wrist RGB-D samples."""
        self._detect_count += 1
        samples = (
            ("head", self._head_rgb, self._head_depth),
            ("left_wrist", self._left_rgb, self._left_depth),
            ("right_wrist", self._right_rgb, self._right_depth),
        )
        best = {}
        bean_candidates = []
        for camera_name, image, depth in samples:
            if image is None or depth is None or self._intrinsics[camera_name] is None:
                continue
            if abs(self._rgb_times[camera_name] - self._depth_times[camera_name]) > self.camera_sync_tolerance:
                continue
            quality = lighting_quality(image)
            self._lighting_status[camera_name] = quality
            if not quality["usable"]:
                if self._detect_count <= 5 or self._detect_count % 10 == 0:
                    self.node.get_logger().warn(
                        f"Skipping {camera_name} frame: unusable lighting {quality}"
                    )
                continue
            for detection in self._run_yolo(
                    self._preprocess(image), camera_name=camera_name, depth=depth):
                name, x, y, z, confidence = detection
                threshold = MIN_CONFIDENCE.get(name, 0.30)
                if confidence < threshold:
                    continue
                if name == "bean":
                    bean_candidates.append((x, y, z, confidence))
                    continue
                previous = best.get(name)
                if previous is None or confidence > previous[4]:
                    best[name] = detection

        detections = list(best.values())
        bean_candidates.sort(key=lambda value: value[3], reverse=True)
        deduplicated_beans = []
        for candidate in bean_candidates:
            point = np.array(candidate[:3])
            if all(np.linalg.norm(point - np.array(existing[:3])) >= 0.02
                   for existing in deduplicated_beans):
                deduplicated_beans.append(candidate)

        with self._lock:
            now = time.time()
            for name, x, y, z, conf in detections:
                self._detections[name] = (x, y, z, conf, now)

            stale = [
                name for name, value in self._detections.items()
                if now - value[4] > self.detection_max_age
            ]
            for name in stale:
                del self._detections[name]
            self._latest_beans = deduplicated_beans

            self._publish_detections()

    def _run_yolo(self, img, camera_name="head", depth=None):
        """Run YOLO inference and return list of (name, x, y, z, conf)."""
        if self.model is None:
            return []

        try:
            results = self.model(img, verbose=False, conf=0.15)
        except Exception as e:
            self.node.get_logger().warn(f"YOLO inference error: {e}")
            return []

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = r.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                if cls_name not in NAME_TO_CLASS:
                    continue
                obj_name = cls_name

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0

                world_point = self._pixel_to_world(
                    u, v, camera_name=camera_name, depth=depth)
                if world_point is None:
                    continue
                wx, wy, wz = world_point

                detections.append((obj_name, wx, wy, wz, conf))

        if self._detect_count <= 5 and detections:
            det_str = ", ".join(
                f"{d[0]}({d[4]:.2f})" for d in detections
            )
            self.node.get_logger().info(f"  YOLO: {det_str}")

        return detections

    def _pixel_to_world(self, u, v, camera_name="head", depth=None):
        """Back-project a valid depth pixel and transform optical frame -> world via TF."""
        intrinsics = self._intrinsics.get(camera_name)
        frame_id = self._camera_frames.get(camera_name, "")
        if depth is None or intrinsics is None or not frame_id:
            return None
        try:
            iv, iu = int(round(v)), int(round(u))
            if not (0 <= iv < depth.shape[0] and 0 <= iu < depth.shape[1]):
                return None
            depth_val = float(depth[iv, iu])
            if depth.dtype == np.uint16:
                depth_val /= 1000.0
            if not math.isfinite(depth_val) or not 0.05 <= depth_val <= 5.0:
                return None
            fx, fy, cx, cy = intrinsics
            camera_point = np.array([
                (u - cx) * depth_val / fx,
                (v - cy) * depth_val / fy,
                depth_val,
            ])
            transform = self._tf_buffer.lookup_transform(
                self.world_frame, frame_id, Time(), timeout=Duration(seconds=0.2))
            t = transform.transform.translation
            q = transform.transform.rotation
            return quaternion_transform_point(
                camera_point,
                (t.x, t.y, t.z),
                (q.x, q.y, q.z, q.w),
            )
        except Exception as exc:
            if self._detect_count <= 3:
                self.node.get_logger().warn(
                    f"TF/depth conversion failed for {camera_name}: {exc}"
                )
            return None

    def _publish_detections(self):
        """Publish current detections to /vision/object_positions."""
        if not _HAS_ROS:
            return

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        names = []
        positions = []

        for name, (x, y, z, conf, ts) in self._detections.items():
            names.append(f"{name}:{conf:.2f}")
            positions.extend([x, y, z])

        msg.name = names
        msg.position = positions
        self._obj_pub.publish(msg)

        bean_msg = JointState()
        bean_msg.header.stamp = msg.header.stamp
        bean_msg.header.frame_id = self.world_frame
        bean_msg.name = [f"bean_{i:04d}" for i in range(len(self._latest_beans))]
        bean_msg.position = []
        for x, y, z, _ in self._latest_beans:
            bean_msg.position.extend([x, y, z])
        self._bean_pub.publish(bean_msg)

        if self._detect_count <= 5 or self._detect_count % 10 == 0:
            det_str = ", ".join(
                f"{n.split(':')[0]}({n.split(':')[1]})" for n in names
            ) if names else "none"
            self.node.get_logger().info(
                f"  Published {len(names)} objects: {det_str}"
            )

    def get_object_pose(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Get latest pose + confidence for an object.

        Returns (x, y, z, conf) or None if not detected.
        NEVER returns hardcoded coordinates.
        """
        with self._lock:
            det = self._detections.get(name)
            if det is None:
                return None
            x, y, z, conf, ts = det
            if time.time() - ts > self.detection_max_age:
                self.node.get_logger().warn(
                    f"  Stale detection for {name} ({time.time()-ts:.1f}s old) — discarding"
                )
                del self._detections[name]
                return None
            return x, y, z, conf

    def _preprocess(self, img):
        """Apply exposure-adaptive gamma and CLAHE before detector inference."""
        try:
            import cv2
            img = img.copy()

            gamma = adaptive_gamma(img)
            lut = np.array([
                ((i / 255.0) ** gamma) * 255 for i in range(256)
            ]).astype(np.uint8)
            img = cv2.LUT(img, lut)

            # CLAHE on L channel
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Light denoise
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=5, hColor=5,
                templateWindowSize=7, searchWindowSize=21
            )
            return img
        except Exception:
            return img

def create_vision_node(robot_mode: str = "sim", config_path: Optional[str] = None,
                       model_path: Optional[str] = None):
    """Create and return a VisionSystem attached to a ROS2 node."""
    if not _HAS_ROS:
        return None

    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("vision_system")
    vision = VisionSystem(
        node, robot_mode=robot_mode, config_path=config_path, model_path=model_path)
    return node, vision


def main():
    parser = argparse.ArgumentParser(description="EBiM Task 3 multi-camera perception")
    parser.add_argument("--robot-mode", choices=("sim", "real"), default="sim")
    parser.add_argument("--config", default=os.environ.get("ROBOT_IO_CONFIG"))
    parser.add_argument("--model-path", default=os.environ.get("VISION_MODEL_PATH"))
    args = parser.parse_args()
    if not _HAS_ROS:
        print("FATAL: ROS2/tf2 is unavailable", file=sys.stderr)
        return 69
    node = None
    try:
        rclpy.init()
        node, _ = create_vision_node(
            robot_mode=args.robot_mode,
            config_path=args.config,
            model_path=args.model_path,
        )
        rclpy.spin(node)
    except ContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 64
    except Exception as exc:
        print(f"FATAL: perception initialization failed: {exc}", file=sys.stderr)
        return 71
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
