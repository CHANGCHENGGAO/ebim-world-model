#!/usr/bin/env python3
"""Fail closed when a Task 3 policy is configured for human teleoperation.

The InnoHub special award permits autonomous command publication through the
simulator bridge, but not keyboard, pedal, or GELLO *human input* during a
run.  This module keeps that distinction explicit and testable.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence


TELEOP_ENVIRONMENT_KEYS = (
    "ENABLE_KEYBOARD_TELEOP",
    "WITH_KEYBOARD_TELEOP",
    "WITH_GELLO_TELEOP",
    "WITH_GELLO_PEDAL_TELEOP",
    "GELLO_ENABLED",
    "KEYBOARD_TELEOP_ENABLED",
    "TELEOP_ADAPTERS",
)
HUMAN_INPUT_TOKENS = ("pedal", "keyboard", "gello")
TELEOP_LAUNCH_TOKENS = ("keyboard", "gello", "teleop")


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off", "none")


def environment_violations(environment: Mapping[str, str]) -> list[str]:
    """Return enabled teleoperation switches and invalid policy selections."""
    violations: list[str] = []
    if environment.get("AUTONOMOUS_ONLY", "1") != "1":
        violations.append("AUTONOMOUS_ONLY must be exactly '1'")
    if environment.get("POLICY_MODE", "closed_loop") != "closed_loop":
        violations.append("POLICY_MODE must be 'closed_loop'")
    for key in TELEOP_ENVIRONMENT_KEYS:
        value = environment.get(key)
        if _enabled(value):
            violations.append(f"teleoperation environment is enabled: {key}={value!r}")
    return violations


class _SubscriptionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.human_input_subscriptions: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        function_name = getattr(node.func, "attr", "")
        if function_name == "create_subscription":
            rendered = " ".join(ast.unparse(argument) for argument in node.args).lower()
            if any(token in rendered for token in HUMAN_INPUT_TOKENS):
                self.human_input_subscriptions.append(rendered)
        self.generic_visit(node)


def policy_subscription_violations(policy_file: str | os.PathLike[str]) -> list[str]:
    """Reject source that subscribes to keyboard, pedal, or GELLO input."""
    path = Path(policy_file)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"cannot inspect policy source {path}: {exc}"]
    visitor = _SubscriptionVisitor()
    visitor.visit(tree)
    return [f"human-input subscription detected: {call}" for call in visitor.human_input_subscriptions]


def launch_violations(launch_file: str | os.PathLike[str]) -> list[str]:
    """Reject actual ROS/Python teleop launch commands, not explanatory text."""
    path = Path(launch_file)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot inspect launch source {path}: {exc}"]
    violations: list[str] = []
    for number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].lower()
        launches_ros = re.search(r"\bros2\s+(?:run|launch)\b", line) is not None
        launches_python = re.search(r"\bpython(?:3)?\b", line) is not None
        if (launches_ros or launches_python) and any(token in line for token in TELEOP_LAUNCH_TOKENS):
            violations.append(f"human-teleop launch detected at {path}:{number}")
    return violations


def validate_autonomy(
    environment: Mapping[str, str],
    policy_file: str | os.PathLike[str],
    launch_file: str | os.PathLike[str] | None = None,
) -> list[str]:
    violations = environment_violations(environment) + policy_subscription_violations(policy_file)
    if launch_file is not None:
        violations += launch_violations(launch_file)
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate autonomous-only Task 3 startup")
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--launch-file", help="shell/Python entrypoint to inspect for teleop launches")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)
    violations = validate_autonomy(os.environ, args.policy_file, args.launch_file)
    report = {"autonomous_only": not violations, "violations": violations}
    if args.json:
        print(json.dumps(report, sort_keys=True))
    elif violations:
        for violation in violations:
            print(f"AUTONOMY_ERROR: {violation}")
    else:
        print("AUTONOMY_OK: no human-input subscription or teleop launch switch found")
    return 0 if not violations else 65


if __name__ == "__main__":
    raise SystemExit(main())
