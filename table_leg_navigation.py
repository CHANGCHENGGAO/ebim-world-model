#!/usr/bin/env python3
"""Geometry-only detection of the two front table legs from a 2-D LiDAR scan.

The detector has no ROS dependency so its cluster/association logic can be
tested before accessing the physical TMR.  It deliberately reports an
observation only; base motion remains fail-closed until the LiDAR mounting and
the vehicle-front offset are calibrated onsite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScanPoint:
    x: float
    y: float
    range_m: float
    index: int = -1


@dataclass(frozen=True)
class LegCluster:
    center_x: float
    center_y: float
    diameter_m: float
    points: tuple[ScanPoint, ...]


@dataclass(frozen=True)
class TableLegPair:
    first: LegCluster
    second: LegCluster
    midpoint_x: float
    midpoint_y: float
    separation_m: float


@dataclass(frozen=True)
class NarrowPassageAssessment:
    """Body-edge clearance estimate relative to a measured centred passage.

    The onsite measurement says a centred robot has about 5 cm on each side.
    A lateral midpoint error consumes that clearance on one side and adds it
    to the other; no unmeasured LiDAR-to-body radial offset is assumed.
    """

    left_clearance_m: float
    right_clearance_m: float
    center_error_m: float
    safe: bool


@dataclass(frozen=True)
class DoorwayCenteringCommand:
    """A bounded base-frame lateral correction inferred from two side ranges.

    Positive ``strafe_m`` means move left in the documented TMR base frame;
    negative means move right.  This deliberately uses relative range balance,
    not a site-specific world/map coordinate.
    """

    left_range_m: float
    right_range_m: float
    center_error_m: float
    strafe_m: float
    centered: bool


def doorway_centering_command(
    left_range_m: float, right_range_m: float, *, center_tolerance_m: float = 0.02,
    max_strafe_step_m: float = 0.04,
) -> DoorwayCenteringCommand:
    """Return one conservative correction from balanced left/right returns.

    If the left side is nearer than the right, the base is left of the passage
    centre and the returned command is negative (move right).  The correction
    is capped so every move is followed by a fresh scan.
    """
    for value, label in ((left_range_m, "left_range_m"),
                         (right_range_m, "right_range_m")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be a finite positive range")
    if center_tolerance_m <= 0.0 or max_strafe_step_m <= 0.0:
        raise ValueError("centering tolerance and maximum step must be positive")
    error = (float(left_range_m) - float(right_range_m)) / 2.0
    centered = abs(error) <= center_tolerance_m
    return DoorwayCenteringCommand(
        float(left_range_m), float(right_range_m), error,
        0.0 if centered else max(-max_strafe_step_m, min(max_strafe_step_m, error)),
        centered,
    )


def assess_narrow_passage(
    pair: TableLegPair, *, nominal_side_clearance_m: float = 0.05,
    minimum_side_clearance_m: float = 0.02, center_tolerance_m: float = 0.02,
) -> NarrowPassageAssessment:
    if not 0.0 < minimum_side_clearance_m < nominal_side_clearance_m:
        raise ValueError("side clearances must satisfy 0 < minimum < nominal")
    if center_tolerance_m <= 0.0:
        raise ValueError("center_tolerance_m must be positive")
    error = float(pair.midpoint_y)
    left = nominal_side_clearance_m + error
    right = nominal_side_clearance_m - error
    safe = (
        abs(error) <= center_tolerance_m
        and left >= minimum_side_clearance_m
        and right >= minimum_side_clearance_m
    )
    return NarrowPassageAssessment(left, right, error, safe)


def stable_narrow_passage_assessment(
    pairs: Sequence[TableLegPair], *, nominal_side_clearance_m: float = 0.05,
    minimum_side_clearance_m: float = 0.02, center_tolerance_m: float = 0.02,
    required_frames: int = 3, maximum_center_spread_m: float = 0.01,
) -> NarrowPassageAssessment | None:
    """Require several consistent scans before allowing a tight passage."""
    if required_frames < 2:
        raise ValueError("required_frames must be at least 2")
    if maximum_center_spread_m <= 0.0:
        raise ValueError("maximum_center_spread_m must be positive")
    recent = tuple(pairs[-required_frames:])
    if len(recent) < required_frames:
        return None
    assessments = tuple(assess_narrow_passage(
        pair,
        nominal_side_clearance_m=nominal_side_clearance_m,
        minimum_side_clearance_m=minimum_side_clearance_m,
        center_tolerance_m=center_tolerance_m,
    ) for pair in recent)
    errors = [item.center_error_m for item in assessments]
    if not all(item.safe for item in assessments):
        return None
    if max(errors) - min(errors) > maximum_center_spread_m:
        return None
    mean_error = sum(errors) / len(errors)
    return NarrowPassageAssessment(
        nominal_side_clearance_m + mean_error,
        nominal_side_clearance_m - mean_error,
        mean_error,
        True,
    )


def scan_to_points(
    ranges: Sequence[float], angle_min: float, angle_increment: float,
    range_min: float, range_max: float, *, sensor_origin_x_m: float = 0.0,
    sensor_origin_y_m: float = 0.0, scan_angle_offset_deg: float = 0.0,
    scan_angle_sign: float = 1.0,
) -> tuple[ScanPoint | None, ...]:
    """Convert a LaserScan into base-frame points using the official URDF TF."""
    if scan_angle_sign not in (-1.0, 1.0):
        raise ValueError("scan_angle_sign must be -1 or +1")
    offset = math.radians(scan_angle_offset_deg)
    points: list[ScanPoint | None] = []
    for index, value in enumerate(ranges):
        distance = float(value)
        if not math.isfinite(distance) or distance < range_min or distance > range_max:
            points.append(None)
            continue
        scan_angle = angle_min + index * angle_increment
        angle = offset + scan_angle_sign * scan_angle
        points.append(ScanPoint(
            x=sensor_origin_x_m + distance * math.cos(angle),
            y=sensor_origin_y_m + distance * math.sin(angle),
            range_m=distance, index=index,
        ))
    return tuple(points)


def _cluster_diameter(points: Sequence[ScanPoint]) -> float:
    return max(
        (math.hypot(first.x - second.x, first.y - second.y)
         for first in points for second in points),
        default=0.0,
    )


def candidate_leg_clusters(
    points: Iterable[ScanPoint | None], *, cluster_gap_m: float = 0.06,
    max_diameter_m: float = 0.16, min_points: int = 2,
) -> tuple[LegCluster, ...]:
    """Cluster adjacent returns and retain compact, forward-facing leg blobs."""
    groups: list[list[ScanPoint]] = []
    current: list[ScanPoint] = []
    previous: ScanPoint | None = None
    for point in points:
        if point is None:
            if current:
                groups.append(current)
            current, previous = [], None
            continue
        if previous is not None and math.hypot(point.x - previous.x, point.y - previous.y) > cluster_gap_m:
            if current:
                groups.append(current)
            current = []
        current.append(point)
        previous = point
    if current:
        groups.append(current)

    candidates: list[LegCluster] = []
    for group in groups:
        if len(group) < min_points:
            continue
        diameter = _cluster_diameter(group)
        center_x = sum(point.x for point in group) / len(group)
        center_y = sum(point.y for point in group) / len(group)
        if center_x <= 0.0 or diameter > max_diameter_m:
            continue
        candidates.append(LegCluster(
            center_x=center_x, center_y=center_y, diameter_m=diameter,
            points=tuple(group),
        ))
    return tuple(candidates)


def detect_table_leg_pair(
    clusters: Sequence[LegCluster], *, separation_min_m: float,
    separation_max_m: float, max_depth_difference_m: float = 0.15,
) -> TableLegPair | None:
    """Select a nearly co-depth, lateral pair of compact leg clusters.

    The closest symmetric pair wins.  This avoids treating a distant wall edge
    as a table, but the resulting pair is still only a candidate until the
    onsite LiDAR mounting/frame calibration confirms it.
    """
    candidates: list[tuple[tuple[float, float, float], TableLegPair]] = []
    for first_index, first in enumerate(clusters):
        for second in clusters[first_index + 1:]:
            separation = math.hypot(first.center_x - second.center_x,
                                    first.center_y - second.center_y)
            depth_difference = abs(first.center_x - second.center_x)
            if not separation_min_m <= separation <= separation_max_m:
                continue
            if depth_difference > max_depth_difference_m:
                continue
            midpoint_x = (first.center_x + second.center_x) / 2.0
            midpoint_y = (first.center_y + second.center_y) / 2.0
            pair = TableLegPair(first, second, midpoint_x, midpoint_y, separation)
            # Prefer a centred pair, then the closest pair, then similar depth.
            candidates.append(((abs(midpoint_y), midpoint_x, depth_difference), pair))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def detect_table_legs_from_scan(
    ranges: Sequence[float], angle_min: float, angle_increment: float,
    range_min: float, range_max: float, *, cluster_gap_m: float = 0.06,
    max_diameter_m: float = 0.16, separation_min_m: float = 0.30,
    separation_max_m: float = 1.40, max_depth_difference_m: float = 0.15,
    sensor_origin_x_m: float = 0.0, sensor_origin_y_m: float = 0.0,
    scan_angle_offset_deg: float = 0.0, scan_angle_sign: float = 1.0,
) -> TableLegPair | None:
    points = scan_to_points(
        ranges, angle_min, angle_increment, range_min, range_max,
        sensor_origin_x_m=sensor_origin_x_m,
        sensor_origin_y_m=sensor_origin_y_m,
        scan_angle_offset_deg=scan_angle_offset_deg,
        scan_angle_sign=scan_angle_sign,
    )
    clusters = candidate_leg_clusters(
        points, cluster_gap_m=cluster_gap_m, max_diameter_m=max_diameter_m)
    return detect_table_leg_pair(
        clusters, separation_min_m=separation_min_m,
        separation_max_m=separation_max_m,
        max_depth_difference_m=max_depth_difference_m)


def desired_leg_range_m(body_front_from_lidar_m: float, body_to_leg_clearance_m: float) -> float:
    """Expected LiDAR-to-leg forward range at the requested body clearance."""
    if body_front_from_lidar_m < 0.0 or body_to_leg_clearance_m <= 0.0:
        raise ValueError("offset and clearance must be positive")
    return body_front_from_lidar_m + body_to_leg_clearance_m


def desired_leg_midpoint_x_m(body_front_x_m: float, body_to_leg_clearance_m: float) -> float:
    """Target table-leg line x in base_link, independent of sensor lateral offset."""
    if body_front_x_m <= 0.0 or body_to_leg_clearance_m <= 0.0:
        raise ValueError("body front extent and clearance must be positive")
    return body_front_x_m + body_to_leg_clearance_m
