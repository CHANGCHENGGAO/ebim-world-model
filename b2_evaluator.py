#!/usr/bin/env python3
"""Independent, controller-score-blind evaluator for B2 episode observations.

Input is an observation export (normally produced from rosbag/video annotation),
not the controller's STAGE_RESULT. Any score fields present in the input are
ignored by construction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


OFFICIAL_OBJECTS = ("plate2", "cup", "bowl2", "spoon2")


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(tuple(float(v) for v in a), tuple(float(v) for v in b))


def point_in_bounds(point: Sequence[float], bounds: Mapping[str, Sequence[float]]) -> bool:
    return all(
        float(bounds[axis][0]) <= float(point[index]) <= float(bounds[axis][1])
        for index, axis in enumerate(("x", "y", "z"))
    )


def score_objects_in_region(
    observations: Mapping[str, Sequence[float]],
    bounds: Mapping[str, Sequence[float]],
    objects: Iterable[str] = OFFICIAL_OBJECTS,
) -> Dict[str, Any]:
    passed = [name for name in objects if name in observations and point_in_bounds(observations[name], bounds)]
    missing_or_outside = [name for name in objects if name not in passed]
    return {"score": min(len(passed), 4), "max_score": 4,
            "passed": passed, "failed": missing_or_outside}


def evaluate_feed(samples: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Dict[str, Any]:
    required_hold = float(spec.get("required_hold_seconds", 3.0))
    tolerance = float(spec.get("feed_zone_tolerance_m", 0.10))
    max_gap = float(spec.get("max_sample_gap_seconds", 0.5))
    max_step = float(spec.get("max_motion_step_m", 1.5))
    ordered = sorted(samples, key=lambda sample: float(sample["timestamp"]))
    continuous = 0.0
    best_hold = 0.0
    current_min_beans = None
    best_min_beans = None
    smooth = True

    for previous, current in zip(ordered, ordered[1:]):
        dt = float(current["timestamp"]) - float(previous["timestamp"])
        step = _distance(previous["spoon_xyz"], current["spoon_xyz"])
        smooth = smooth and step <= max_step
        in_zone = _distance(current["spoon_xyz"], current["feed_target_xyz"]) <= tolerance
        beans = int(current.get("beans_on_spoon", 0))
        if 0.0 < dt <= max_gap and in_zone and beans > 0 and smooth:
            continuous += dt
            current_min_beans = (
                beans if current_min_beans is None else min(current_min_beans, beans)
            )
            if continuous > best_hold:
                best_hold = continuous
                best_min_beans = current_min_beans
        else:
            continuous = 0.0
            current_min_beans = None

    completed = smooth and best_hold >= required_hold and (best_min_beans or 0) > 0
    score = min(int(best_min_beans or 0), 4) if completed else 0
    return {"score": score, "max_score": 4, "completed": completed,
            "continuous_hold_seconds": best_hold, "smooth": smooth,
            "minimum_beans_during_hold": int(best_min_beans or 0)}


def evaluate_recovery(stage: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = int(stage.get("beans_before", 0))
    positions = stage.get("bean_positions_after", [])
    center = spec["container_center_xyz"]
    radius = float(spec["container_radius_m"])
    inside = sum(1 for point in positions if _distance(point, center) <= radius)
    ratio = inside / baseline if baseline > 0 else 0.0
    if ratio >= 1.0:
        score = 4
    elif ratio >= 0.9:
        score = 3
    elif ratio >= 0.8:
        score = 2
    else:
        score = 0
    return {"score": score, "max_score": 4, "beans_inside": inside,
            "beans_before": baseline, "recovery_percent": ratio * 100.0,
            "observable": baseline > 0 and bool(positions)}


def evaluate_episode(episode: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    stage1 = score_objects_in_region(
        episode.get("stage1_final_objects", {}), spec["dining_bounds"])
    stage2 = evaluate_feed(episode.get("feed_samples", []), spec.get("feeding", {}))
    stage3 = evaluate_recovery(episode.get("stage3", {}), spec["recovery"])
    stage4 = score_objects_in_region(
        episode.get("stage4_final_objects", {}), spec["sink_bounds"])
    stages = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "stage4": stage4}
    return {"stages": stages, "total_score": sum(value["score"] for value in stages.values()),
            "max_score": 16, "source": "independent_observations"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an EBiM B2 observation export")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    episode = json.loads(Path(args.episode).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    print(json.dumps(evaluate_episode(episode, spec), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
