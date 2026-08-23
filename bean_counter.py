#!/usr/bin/env python3
"""
Bean Counter for EBiM Task 3 — queries Isaac Sim stage for bean positions.

Uses the same approach as the official evaluation:
  1. Find all Bean_* prims in the stage
  2. Get their world positions via UsdGeom
  3. Count beans inside the recovery container's sphere region

This is ground-truth accurate — no camera image processing needed.
"""

import math
import sys
from typing import List, Tuple, Optional

import numpy as np


def sorted_bean_paths() -> List[str]:
    """Find all Bean_* prim paths in the Isaac Sim stage."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return []

        bean_paths = []
        for prim in Usd.PrimRange(stage):
            name = prim.GetName()
            if name.startswith("Bean_") or name.startswith("bean_"):
                bean_paths.append(str(prim.GetPath()))

        bean_paths.sort()
        return bean_paths
    except Exception as e:
        print(f"[bean_counter] sorted_bean_paths failed: {e}", file=sys.stderr)
        return []


def get_bean_positions() -> List[np.ndarray]:
    """Get world positions of all beans in the scene."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return []

        paths = sorted_bean_paths()
        positions = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            world_transform = xform.ComputeLocalToWorldTransform(0.0)
            translation = world_transform.ExtractTranslation()
            positions.append(np.array([
                translation[0],
                translation[1],
                translation[2],
            ]))
        return positions
    except Exception as e:
        print(f"[bean_counter] get_bean_positions failed: {e}", file=sys.stderr)
        return []


def count_beans_in_sphere(
    center: np.ndarray,
    radius: float,
) -> Tuple[int, int]:
    """Count beans inside a sphere region.

    Args:
        center: sphere center in world coordinates (x, y, z)
        radius: sphere radius in meters

    Returns:
        (beans_inside, beans_total)
    """
    positions = get_bean_positions()
    total = len(positions)
    if total == 0:
        return 0, 0

    inside = 0
    for pos in positions:
        dist = np.linalg.norm(pos - center)
        if dist <= radius:
            inside += 1

    return inside, total


def count_beans_near_container(
    container_pos: Tuple[float, float, float],
    container_size: float = 0.15,
    container_height: float = 0.10,
) -> Tuple[int, int]:
    """Count beans inside or near the recovery container.

    Uses a sphere region matching the official evaluation's
    recovery_region_from_bounds() logic:
      radius = 0.75 * diagonal
      center = container center

    Args:
        container_pos: container center (x, y, z) in world coordinates
        container_size: approximate half-extent of the container opening
        container_height: container height above table

    Returns:
        (beans_inside, beans_total)
    """
    # Diagonal of the container opening (assuming square opening)
    diagonal = math.sqrt(2) * container_size * 2
    # Official formula: radius = 0.75 * diagonal
    radius = 0.75 * diagonal

    center = np.array(container_pos)
    # Raise center to account for container walls
    center[2] += container_height * 0.5

    return count_beans_in_sphere(center, radius)


def count_beans_above_table(
    table_z: float,
    near_pos: Optional[Tuple[float, float, float]] = None,
    xy_radius: float = 0.5,
) -> Tuple[int, int]:
    """Count beans above table surface, optionally near a position.

    Args:
        table_z: table surface z-height
        near_pos: if given, only count beans within xy_radius of this position
        xy_radius: xy distance threshold

    Returns:
        (beans_above, beans_total)
    """
    positions = get_bean_positions()
    total = len(positions)
    if total == 0:
        return 0, 0

    above = 0
    for pos in positions:
        if pos[2] > table_z - 0.01:  # bean is on or above table
            if near_pos is not None:
                dx = pos[0] - near_pos[0]
                dy = pos[1] - near_pos[1]
                if math.sqrt(dx * dx + dy * dy) > xy_radius:
                    continue
            above += 1

    return above, total


def bean_transfer_score(
    container_pos: Tuple[float, float, float],
    container_half_size: float = 0.08,
    container_height: float = 0.12,
) -> Tuple[int, int, int, float]:
    """Calculate bean transfer score for Stage 3.

    Args:
        container_pos: container center position (x, y, z)
        container_half_size: half-extent of container opening
        container_height: container wall height

    Returns:
        (beans_inside, beans_total, score, percentage)
    """
    beans_inside, beans_total = count_beans_near_container(
        container_pos,
        container_size=container_half_size,
        container_height=container_height,
    )

    if beans_total <= 0:
        return 0, 0, 0, 0.0

    ratio = max(0.0, min(1.0, beans_inside / beans_total))
    percentage = ratio * 100.0

    if ratio >= 1.0:
        score = 4
    elif ratio >= 0.9:
        score = 3
    elif ratio >= 0.8:
        score = 2
    else:
        score = 0

    return beans_inside, beans_total, score, percentage
