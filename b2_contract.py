#!/usr/bin/env python3
"""Pure-Python configuration and freshness helpers for the B2 policy.

This module deliberately has no ROS dependency so the I/O contract and safety
rules can be validated in CI and before a robot connection is available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from navigation_safety import parse_onsite_bowl_cup_route


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "robot_io.json"
SUPPORTED_SCHEMA_VERSION = 1

# Canonical task vocabulary used end-to-end by perception and the controller.
# Legacy bridge names are accepted at the input boundary only; no policy code
# should query them or publish them.
CANONICAL_OBJECT_NAMES = frozenset((
    "plate", "cup", "bowl", "spoon", "head", "sink",
    "recycling_bin", "seat_area", "simple_tray", "bean",
))
LEGACY_OBJECT_ALIASES = {
    "plate2": "plate",
    "bowl2": "bowl",
    "spoon2": "spoon",
    "ikea_knock_box": "recycling_bin",
}


class ContractError(ValueError):
    """Raised when a robot I/O contract is incomplete or unsafe."""


@dataclass(frozen=True)
class DetectionObservation:
    position: Tuple[float, float, float]
    confidence: float
    frame_id: str
    source_stamp: float
    received_at: float

    def is_fresh(self, now: float, max_age: float) -> bool:
        return max_age > 0.0 and 0.0 <= now - self.received_at <= max_age


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _require_nav_targets(value: Any, label: str) -> Dict[str, Sequence[float]]:
    targets = _require_mapping(value, label)
    validated: Dict[str, Sequence[float]] = {}
    for key in ("home", "kitchen", "sink"):
        pose = targets.get(key)
        if (not isinstance(pose, list) or len(pose) != 3 or
                not all(isinstance(number, (int, float)) for number in pose)):
            raise ContractError(f"{label}.{key} must be [x, y, yaw_deg]")
        validated[key] = [float(number) for number in pose]
    return validated


def load_contract(path: str | os.PathLike[str] | None, mode: str) -> Dict[str, Any]:
    """Load and strictly validate one sim/real configuration section."""
    if mode not in ("sim", "real"):
        raise ContractError(f"unsupported robot mode: {mode!r}")

    config_path = Path(path or os.environ.get("ROBOT_IO_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_file():
        raise ContractError(f"robot I/O config not found: {config_path}")

    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read robot I/O config {config_path}: {exc}") from exc

    root = _require_mapping(document, "root")
    if root.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ContractError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}, "
            f"got {root.get('schema_version')!r}"
        )
    modes = _require_mapping(root.get("modes"), "modes")
    selected = dict(_require_mapping(modes.get(mode), f"modes.{mode}"))
    topics = dict(_require_mapping(selected.get("topics"), f"modes.{mode}.topics"))

    required_topic_keys = tuple(selected.get("required_topic_keys", ()))
    if not required_topic_keys or not all(isinstance(k, str) for k in required_topic_keys):
        raise ContractError(f"modes.{mode}.required_topic_keys must be a non-empty string list")
    for key in required_topic_keys:
        _require_string(topics, key, f"modes.{mode}.topics")
    static_required_topic_keys = tuple(selected.get("static_required_topic_keys", ()))
    if not all(isinstance(key, str) for key in static_required_topic_keys):
        raise ContractError(
            f"modes.{mode}.static_required_topic_keys must be a string list")
    unknown_static_keys = set(static_required_topic_keys) - set(required_topic_keys)
    if unknown_static_keys:
        raise ContractError(
            f"modes.{mode}.static_required_topic_keys must be required topics: "
            f"{sorted(unknown_static_keys)}")

    for key in (
        "left_arm_cmd", "right_arm_cmd", "left_arm_state", "right_arm_state",
        "left_gripper_cmd", "right_gripper_cmd", "left_gripper_state",
        "right_gripper_state", "base_cmd", "left_wrench", "right_wrench",
        "odom", "vision_objects", "vision_beans", "head_rgb",
        "left_wrist_rgb", "right_wrist_rgb", "front_lidar", "rear_lidar",
    ):
        _require_string(topics, key, f"modes.{mode}.topics")

    base_command_type = selected.get("base_command_type")
    if base_command_type not in ("pedal_string", "twist", "twist_stamped"):
        raise ContractError(
            f"modes.{mode}.base_command_type must be pedal_string, twist, or twist_stamped")
    if base_command_type == "pedal_string":
        _require_string(topics, "pedal_cmd", f"modes.{mode}.topics")

    vision_3d_available = selected.get("vision_3d_available")
    if not isinstance(vision_3d_available, bool):
        raise ContractError(f"modes.{mode}.vision_3d_available must be boolean")
    if vision_3d_available:
        for key in (
            "head_depth", "head_camera_info", "left_wrist_depth",
            "left_wrist_camera_info", "right_wrist_depth", "right_wrist_camera_info",
        ):
            _require_string(topics, key, f"modes.{mode}.topics")

    selected["topics"] = topics
    selected["world_frame"] = _require_string(selected, "world_frame", f"modes.{mode}")
    selected["nav_targets"] = _require_nav_targets(
        selected.get("nav_targets"), f"modes.{mode}.nav_targets")
    try:
        # This route is intentionally optional for the official full Task 3
        # workflow.  If it is enabled, the parser requires real, bounded
        # calibration values instead of accepting placeholder coordinates.
        route = parse_onsite_bowl_cup_route(
            selected.get("onsite_bowl_cup_route", {
                "enabled": False, "bowl_arm": "right", "cup_arm": "left",
            }))
    except ValueError as exc:
        raise ContractError(f"modes.{mode}.onsite_bowl_cup_route invalid: {exc}") from exc
    selected["onsite_bowl_cup_route"] = selected.get("onsite_bowl_cup_route", {
        "enabled": route.enabled, "bowl_arm": route.bowl_arm, "cup_arm": route.cup_arm,
        "minimum_lidar_clearance_m": route.minimum_lidar_clearance_m,
    })
    limits = dict(_require_mapping(selected.get("limits"), f"modes.{mode}.limits"))
    force_limit = limits.get("force_stop_threshold_n")
    if not isinstance(force_limit, (int, float)) or force_limit <= 0:
        raise ContractError(
            f"modes.{mode}.limits.force_stop_threshold_n must be positive")
    limits["force_stop_threshold_n"] = float(force_limit)
    selected["limits"] = limits
    obstacle_safety_required = selected.get("obstacle_safety_required")
    if not isinstance(obstacle_safety_required, bool):
        raise ContractError(f"modes.{mode}.obstacle_safety_required must be boolean")
    for key in (
        "gripper_command_open", "gripper_command_closed",
        "gripper_feedback_open", "gripper_feedback_closed",
    ):
        value = selected.get(key)
        if not isinstance(value, (int, float)):
            raise ContractError(f"modes.{mode}.{key} must be numeric")
        selected[key] = float(value)
    if selected["gripper_command_open"] == selected["gripper_command_closed"]:
        raise ContractError(f"modes.{mode} gripper command endpoints must differ")
    if selected["gripper_feedback_open"] == selected["gripper_feedback_closed"]:
        raise ContractError(f"modes.{mode} gripper feedback endpoints must differ")
    selected["required_topic_keys"] = list(required_topic_keys)
    selected["static_required_topic_keys"] = list(static_required_topic_keys)
    selected["config_path"] = str(config_path)
    selected["mode"] = mode
    return selected


def required_topics(contract: Mapping[str, Any]) -> Sequence[str]:
    topics = _require_mapping(contract.get("topics"), "contract.topics")
    return tuple(str(topics[key]) for key in contract["required_topic_keys"])


def parse_detection_name(value: str) -> Tuple[str, float]:
    """Parse `object:confidence`; reject malformed/non-finite confidence."""
    name, separator, raw_confidence = value.rpartition(":")
    if not separator:
        return value, 0.0
    try:
        confidence = float(raw_confidence)
    except ValueError:
        return value, 0.0
    if not 0.0 <= confidence <= 1.0:
        return name or value, 0.0
    return name, confidence


def normalize_object_name(value: str) -> str | None:
    """Return a canonical task object name, or None for an unknown label."""
    candidate = LEGACY_OBJECT_ALIASES.get(value.strip(), value.strip())
    return candidate if candidate in CANONICAL_OBJECT_NAMES else None


def stamp_to_seconds(stamp: Any) -> float:
    sec = float(getattr(stamp, "sec", 0.0))
    nanosec = float(getattr(stamp, "nanosec", 0.0))
    return sec + nanosec / 1_000_000_000.0


def stale_keys(
    required_keys: Iterable[str],
    last_seen: Mapping[str, float],
    now: float | None,
    max_age: float,
    static_keys: Iterable[str] = (),
    max_age_by_key: Mapping[str, float] | None = None,
) -> Sequence[str]:
    current = time.monotonic() if now is None else now
    static = frozenset(static_keys)
    overrides = max_age_by_key or {}
    return tuple(
        key for key in required_keys
        if key not in last_seen or (
            key not in static and
            current - last_seen[key] > float(overrides.get(key, max_age)))
    )


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> Tuple[float, float, float]:
    """Apply a validated 4x4 homogeneous transform without numpy."""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ContractError("camera_to_world must be a 4x4 matrix")
    if len(point) != 3:
        raise ContractError("point must contain x, y, z")
    vector = (float(point[0]), float(point[1]), float(point[2]), 1.0)
    out = tuple(sum(float(matrix[r][c]) * vector[c] for c in range(4)) for r in range(4))
    if abs(out[3]) < 1e-12:
        raise ContractError("homogeneous transform produced w=0")
    return out[0] / out[3], out[1] / out[3], out[2] / out[3]


def quaternion_transform_point(
    point: Sequence[float],
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> Tuple[float, float, float]:
    """Apply a normalized quaternion rotation followed by translation."""
    if len(point) != 3 or len(translation) != 3 or len(quaternion_xyzw) != 4:
        raise ContractError("point/translation/quaternion sizes must be 3/3/4")
    px, py, pz = (float(v) for v in point)
    tx, ty, tz = (float(v) for v in translation)
    qx, qy, qz, qw = (float(v) for v in quaternion_xyzw)
    norm = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    if norm < 1e-12:
        raise ContractError("quaternion norm is zero")
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    # Quaternion-derived rotation matrix.
    rx = (1 - 2*(qy*qy + qz*qz))*px + 2*(qx*qy - qz*qw)*py + 2*(qx*qz + qy*qw)*pz
    ry = 2*(qx*qy + qz*qw)*px + (1 - 2*(qx*qx + qz*qz))*py + 2*(qy*qz - qx*qw)*pz
    rz = 2*(qx*qz - qy*qw)*px + 2*(qy*qz + qx*qw)*py + (1 - 2*(qx*qx + qy*qy))*pz
    return rx + tx, ry + ty, rz + tz


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate/inspect the EBiM B2 robot I/O contract")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mode", choices=("sim", "real"), required=True)
    parser.add_argument("--print-required-topics", action="store_true")
    parser.add_argument("--allow-unconfirmed", action="store_true")
    args = parser.parse_args()

    try:
        contract = load_contract(args.config, args.mode)
        if not contract.get("confirmed", False) and not args.allow_unconfirmed:
            raise ContractError(
                f"{args.mode} interface is marked UNCONFIRMED; update the config from the "
                "official contract or pass --allow-unconfirmed only for bench testing"
            )
        if args.print_required_topics:
            print("\n".join(required_topics(contract)))
        else:
            print(json.dumps(contract, indent=2, sort_keys=True))
    except ContractError as exc:
        print(f"CONTRACT_ERROR: {exc}", file=os.sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
