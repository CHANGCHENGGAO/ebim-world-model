#!/usr/bin/env python3
"""Pure-Python ground-obstacle path safety contract for the InnoHub award.

The organizer's `plus` obstacle message is not public yet.  This module keeps
the geometry and fail-closed planning independent of that future adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


Point2D = tuple[float, float]
ALLOWED_OBSTACLES = frozenset(("cable", "book", "ball"))
ALLOWED_RELATIVE_AXES = frozenset(("forward", "strafe", "rotate"))


def wrap_degrees(angle: float) -> float:
    """Return the shortest signed angular displacement in [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def directed_progress(moved: float, target: float) -> float:
    """Progress in the requested direction; negative means moving backwards."""
    if target == 0.0:
        return 0.0
    return float(moved) if target > 0.0 else -float(moved)


def minimum_range_in_sector(
    ranges: Sequence[float], angle_min: float, angle_increment: float,
    range_min: float, range_max: float, *, center_angle_deg: float = 0.0,
    half_angle_deg: float = 25.0,
) -> float | None:
    """Nearest valid return in a named motion corridor of a LaserScan."""
    half_angle = math.radians(float(half_angle_deg))
    center_angle = math.radians(float(center_angle_deg))
    valid: list[float] = []
    for index, raw in enumerate(ranges):
        distance = float(raw)
        if not math.isfinite(distance) or not range_min <= distance <= range_max:
            continue
        angle_error = math.atan2(
            math.sin(angle_min + index * angle_increment - center_angle),
            math.cos(angle_min + index * angle_increment - center_angle),
        )
        if abs(angle_error) <= half_angle:
            valid.append(distance)
    return min(valid) if valid else None


def tapered_navigation_speed(axis: str, nominal_speed: float, remaining: float) -> float:
    """Two-stage approach speed derived from low-speed real-base coasting."""
    if axis not in ("forward", "strafe", "rotation"):
        raise ValueError(f"unsupported navigation axis: {axis}")
    if nominal_speed <= 0.0 or remaining < 0.0:
        raise ValueError("nominal_speed must be positive and remaining non-negative")
    if axis == "rotation":
        if remaining <= 3.0:
            return max(1.0, nominal_speed * 0.25)
        if remaining <= 10.0:
            return max(2.0, nominal_speed * 0.50)
    else:
        if remaining <= 0.06:
            return max(0.01, nominal_speed * 0.25)
        if remaining <= 0.15:
            return max(0.015, nominal_speed * 0.50)
    return nominal_speed


@dataclass(frozen=True)
class GroundObstacle:
    kind: str
    center: Point2D
    radius_m: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_OBSTACLES:
            raise ValueError(f"unsupported ground obstacle: {self.kind!r}")
        if self.radius_m <= 0.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("radius_m must be positive and confidence must be in [0, 1]")


@dataclass(frozen=True)
class SafePath:
    waypoints: tuple[Point2D, ...]
    reason: str

    @property
    def safe(self) -> bool:
        return bool(self.waypoints)


@dataclass(frozen=True)
class RelativeMotion:
    """One odometry-closed relative base motion for the onsite workflow.

    Linear distances are metres; rotation distances are degrees.  The route is
    deliberately relative to the measured odometry at workflow start, so it
    does not silently depend on an unverified global map/TF origin.
    """

    name: str
    axis: str
    distance: float
    speed: float
    narrow_passage: bool = False


@dataclass(frozen=True)
class OnsiteBowlCupRoute:
    """Validated calibration contract for the limited onsite bowl/cup route."""

    enabled: bool
    bowl_arm: str
    cup_arm: str
    minimum_lidar_clearance_m: float
    narrow_nominal_side_clearance_m: float
    narrow_minimum_side_clearance_m: float
    narrow_center_tolerance_m: float
    recovery_attempts: int
    recovery_wait_seconds: float
    base_settle_seconds: float
    start_to_kitchen: tuple[RelativeMotion, ...]
    dining_transition: tuple[RelativeMotion, ...]

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.start_to_kitchen) and bool(self.dining_transition)


def _positive_number(value: Any, label: str, *, lower: float, upper: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    numeric = float(value)
    if not lower <= numeric <= upper:
        raise ValueError(f"{label} must be within [{lower}, {upper}]")
    return numeric


def _nonzero_number(value: Any, label: str, *, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    numeric = float(value)
    if not 0.01 <= abs(numeric) <= maximum:
        raise ValueError(f"{label} magnitude must be within [0.01, {maximum}]")
    return numeric


def _parse_relative_leg(value: Any, label: str) -> tuple[RelativeMotion, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    motions: list[RelativeMotion] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{item_label} must be an object")
        name = raw.get("name")
        axis = raw.get("axis")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{item_label}.name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"{label} contains duplicate motion name {name!r}")
        seen_names.add(name)
        if axis not in ALLOWED_RELATIVE_AXES:
            raise ValueError(
                f"{item_label}.axis must be one of {sorted(ALLOWED_RELATIVE_AXES)}")
        # A single primitive longer than 2.5m is intentionally rejected.  It
        # forces the narrow route to be observed and checked in short legs.
        # The measured direction is site-dependent.  Signed values avoid a
        # code edit onsite: positive is controller FWD/strafe-A/rotate-A+C;
        # negative reverses that command.  The return route still inverts it.
        distance = _nonzero_number(raw.get("distance"), f"{item_label}.distance",
                                   maximum=2.5 if axis != "rotate" else 90.0)
        speed = _positive_number(raw.get("speed"), f"{item_label}.speed",
                                 lower=0.03 if axis != "rotate" else 3.0,
                                 upper=0.25 if axis != "rotate" else 30.0)
        narrow = raw.get("narrow_passage", False)
        if not isinstance(narrow, bool):
            raise ValueError(f"{item_label}.narrow_passage must be boolean")
        if narrow and axis != "forward":
            raise ValueError(f"{item_label}.narrow_passage is only valid for forward motion")
        motions.append(RelativeMotion(name.strip(), axis, distance, speed, narrow))
    return tuple(motions)


def parse_onsite_bowl_cup_route(value: Any) -> OnsiteBowlCupRoute:
    """Parse the fail-closed, calibration-first onsite navigation route.

    A disabled route is valid so the repository can ship without invented
    site dimensions.  Once enabled, every movement value must be present and
    bounded before the controller can publish a base command.
    """
    if not isinstance(value, Mapping):
        raise ValueError("onsite_bowl_cup_route must be an object")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("onsite_bowl_cup_route.enabled must be boolean")
    bowl_arm = value.get("bowl_arm")
    cup_arm = value.get("cup_arm")
    if bowl_arm not in ("left", "right") or cup_arm not in ("left", "right"):
        raise ValueError("onsite_bowl_cup_route bowl_arm/cup_arm must be left or right")
    if bowl_arm == cup_arm:
        raise ValueError("onsite_bowl_cup_route bowl_arm and cup_arm must differ")
    clearance = _positive_number(
        value.get("minimum_lidar_clearance_m", 0.75),
        "onsite_bowl_cup_route.minimum_lidar_clearance_m", lower=0.40, upper=1.50)
    narrow_nominal = _positive_number(
        value.get("narrow_nominal_side_clearance_m", 0.05),
        "onsite_bowl_cup_route.narrow_nominal_side_clearance_m", lower=0.03, upper=0.20)
    narrow_minimum = _positive_number(
        value.get("narrow_minimum_side_clearance_m", 0.02),
        "onsite_bowl_cup_route.narrow_minimum_side_clearance_m", lower=0.01, upper=0.10)
    narrow_tolerance = _positive_number(
        value.get("narrow_center_tolerance_m", 0.02),
        "onsite_bowl_cup_route.narrow_center_tolerance_m", lower=0.005, upper=0.10)
    if narrow_minimum >= narrow_nominal:
        raise ValueError("narrow minimum side clearance must be below nominal clearance")
    if narrow_tolerance > narrow_nominal - narrow_minimum:
        raise ValueError("narrow center tolerance would violate minimum side clearance")
    recovery_attempts = value.get("recovery_attempts", 15)
    if (not isinstance(recovery_attempts, int) or isinstance(recovery_attempts, bool)
            or not 1 <= recovery_attempts <= 30):
        raise ValueError("onsite_bowl_cup_route.recovery_attempts must be within [1, 30]")
    recovery_wait = _positive_number(
        value.get("recovery_wait_seconds", 2.0),
        "onsite_bowl_cup_route.recovery_wait_seconds", lower=0.5, upper=10.0)
    base_settle = _positive_number(
        value.get("base_settle_seconds", 3.0),
        "onsite_bowl_cup_route.base_settle_seconds", lower=0.2, upper=10.0)
    if not enabled:
        return OnsiteBowlCupRoute(
            False, bowl_arm, cup_arm, clearance, narrow_nominal,
            narrow_minimum, narrow_tolerance, recovery_attempts, recovery_wait,
            base_settle, (), ())
    start_to_kitchen = _parse_relative_leg(
        value.get("start_to_kitchen"), "onsite_bowl_cup_route.start_to_kitchen")
    dining_transition = _parse_relative_leg(
        value.get("dining_transition"), "onsite_bowl_cup_route.dining_transition")
    if not any(motion.narrow_passage for motion in start_to_kitchen):
        raise ValueError("onsite_bowl_cup_route.start_to_kitchen needs one narrow_passage motion")
    return OnsiteBowlCupRoute(
        True, bowl_arm, cup_arm, clearance, narrow_nominal, narrow_minimum,
        narrow_tolerance, recovery_attempts, recovery_wait,
        base_settle, start_to_kitchen, dining_transition)


def _distance_point_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    sx, sy = start
    ex, ey = end
    px, py = point
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - sx, py - sy)
    scale = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    return math.hypot(px - (sx + scale * dx), py - (sy + scale * dy))


def segment_is_clear(start: Point2D, end: Point2D, obstacles: Iterable[GroundObstacle], clearance_m: float) -> bool:
    return all(
        _distance_point_to_segment(obstacle.center, start, end)
        > obstacle.radius_m + clearance_m
        for obstacle in obstacles
    )


def plan_safe_path(
    start: Point2D,
    goal: Point2D,
    obstacles: Sequence[GroundObstacle] | None,
    *,
    robot_radius_m: float = 0.40,
    safety_margin_m: float = 0.20,
) -> SafePath:
    """Return a direct or single-detour path; unknown obstacles fail closed."""
    if obstacles is None:
        return SafePath((), "ground obstacle observation unavailable")
    clearance = robot_radius_m + safety_margin_m
    if segment_is_clear(start, goal, obstacles, clearance):
        return SafePath((goal,), "direct path clear")

    dx, dy = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return SafePath((goal,), "already at goal")
    normal = (-dy / length, dx / length)
    blocking = [
        obstacle for obstacle in obstacles
        if _distance_point_to_segment(obstacle.center, start, goal)
        <= obstacle.radius_m + clearance
    ]
    candidates: list[Point2D] = []
    for sign in (-1.0, 1.0):
        # A midpoint detour creates two diagonal segments.  Give it additional
        # lateral room so both segments, not just the waypoint, clear the
        # inflated obstacle footprint.
        offset = 1.60 * max(obstacle.radius_m + clearance for obstacle in blocking)
        midpoint = ((start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0)
        candidates.append((midpoint[0] + sign * normal[0] * offset,
                           midpoint[1] + sign * normal[1] * offset))
    for detour in candidates:
        if (segment_is_clear(start, detour, obstacles, clearance) and
                segment_is_clear(detour, goal, obstacles, clearance)):
            return SafePath((detour, goal), "single-obstacle detour")
    return SafePath((), "no verified collision-free path")
