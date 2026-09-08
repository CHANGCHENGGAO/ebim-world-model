import math
import os
import tempfile
import unittest
from pathlib import Path

from autonomy_guard import environment_violations, launch_violations, policy_subscription_violations
from navigation_safety import (
    GroundObstacle,
    parse_onsite_bowl_cup_route,
    plan_safe_path,
    segment_is_clear,
    directed_progress,
    minimum_range_in_sector,
    tapered_navigation_speed,
    wrap_degrees,
)
from onsite_preflight import EXPECTED_REAL_PLATFORM_TYPES, parse_topic_list_t, validate_topic_types
from robustness_matrix import EvidenceError, score_complete_robustness, score_robustness
from table_leg_navigation import (
    LegCluster,
    ScanPoint,
    detect_table_leg_pair,
    desired_leg_range_m,
    desired_leg_midpoint_x_m,
    assess_narrow_passage,
    scan_to_points,
    stable_narrow_passage_assessment,
)
from vision_callback import adaptive_gamma, lighting_quality


class InnoHubRobustnessTests(unittest.TestCase):
    def test_table_leg_pair_geometry_and_body_clearance(self):
        left = LegCluster(1.20, 0.42, 0.06, (
            ScanPoint(1.20, 0.39, 1.26), ScanPoint(1.20, 0.45, 1.28)))
        right = LegCluster(1.22, -0.42, 0.06, (
            ScanPoint(1.22, -0.45, 1.30), ScanPoint(1.22, -0.39, 1.28)))
        pair = detect_table_leg_pair(
            (left, right), separation_min_m=0.50, separation_max_m=1.10)
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair.midpoint_y, 0.0, places=7)
        self.assertAlmostEqual(pair.separation_m, 0.84, places=2)
        self.assertAlmostEqual(desired_leg_range_m(0.27, 0.13), 0.40)
        self.assertAlmostEqual(desired_leg_midpoint_x_m(0.40, 0.16), 0.56)

    def test_official_front_lidar_transform_to_base_link(self):
        # Front LiDAR scan +45 degrees is robot forward in the official TMR URDF.
        points = scan_to_points(
            (1.0,), math.radians(45.0), 0.0, 0.05, 10.0,
            sensor_origin_x_m=0.3275, sensor_origin_y_m=0.2175,
            scan_angle_offset_deg=45.0, scan_angle_sign=-1.0)
        self.assertAlmostEqual(points[0].x, 1.3275, places=4)
        self.assertAlmostEqual(points[0].y, 0.2175, places=4)

    def test_narrow_passage_uses_measured_body_edge_clearance(self):
        left = LegCluster(1.0, 0.42, 0.05, ())
        right = LegCluster(1.0, -0.38, 0.05, ())
        pair = detect_table_leg_pair(
            (left, right), separation_min_m=0.50, separation_max_m=1.10)
        assessment = assess_narrow_passage(pair)
        self.assertTrue(assessment.safe)
        self.assertAlmostEqual(assessment.left_clearance_m, 0.07)
        self.assertAlmostEqual(assessment.right_clearance_m, 0.03)

        badly_offset = detect_table_leg_pair(
            (LegCluster(1.0, 0.46, 0.05, ()),
             LegCluster(1.0, -0.34, 0.05, ())),
            separation_min_m=0.50, separation_max_m=1.10)
        self.assertFalse(assess_narrow_passage(badly_offset).safe)

    def test_narrow_passage_requires_three_consistent_frames(self):
        pairs = []
        for error in (0.004, 0.006, 0.005):
            pairs.append(detect_table_leg_pair(
                (LegCluster(1.0, 0.40 + error, 0.05, ()),
                 LegCluster(1.0, -0.40 + error, 0.05, ())),
                separation_min_m=0.50, separation_max_m=1.10))
        self.assertIsNone(stable_narrow_passage_assessment(pairs[:2]))
        stable = stable_narrow_passage_assessment(pairs)
        self.assertIsNotNone(stable)
        self.assertAlmostEqual(stable.center_error_m, 0.005, places=3)
        noisy = pairs[:2] + [detect_table_leg_pair(
            (LegCluster(1.0, 0.43, 0.05, ()),
             LegCluster(1.0, -0.37, 0.05, ())),
            separation_min_m=0.50, separation_max_m=1.10)]
        self.assertIsNone(stable_narrow_passage_assessment(noisy))

    def test_lidar_uses_motion_sector_not_side_wall(self):
        # -90, -45, 0, +45, +90 degrees: side returns must not stop forward motion.
        nearest = minimum_range_in_sector(
            (0.05, 0.10, 0.80, 0.10, 0.05), -math.pi / 2, math.pi / 4,
            0.02, 10.0, half_angle_deg=25.0)
        self.assertAlmostEqual(nearest, 0.80)

        official_forward = minimum_range_in_sector(
            (0.10, 0.80, 0.10), 0.0, math.pi / 4,
            0.02, 10.0, center_angle_deg=45.0, half_angle_deg=20.0)
        self.assertAlmostEqual(official_forward, 0.80)

    def test_odom_progress_handles_direction_and_yaw_wrap(self):
        self.assertAlmostEqual(wrap_degrees(-358.0), 2.0)
        self.assertAlmostEqual(wrap_degrees(358.0), -2.0)
        self.assertGreater(directed_progress(-0.2, -1.0), 0.0)
        self.assertLess(directed_progress(0.2, -1.0), 0.0)

    def test_real_base_approach_speed_tapers_near_target(self):
        self.assertAlmostEqual(tapered_navigation_speed("forward", 0.08, 0.20), 0.08)
        self.assertAlmostEqual(tapered_navigation_speed("forward", 0.08, 0.10), 0.04)
        self.assertAlmostEqual(tapered_navigation_speed("forward", 0.08, 0.04), 0.02)
        self.assertAlmostEqual(tapered_navigation_speed("rotation", 12.0, 8.0), 6.0)
        self.assertAlmostEqual(tapered_navigation_speed("rotation", 12.0, 2.0), 3.0)

    def test_onsite_route_is_fail_closed_until_calibrated(self):
        route = parse_onsite_bowl_cup_route({
            "enabled": False,
            "bowl_arm": "right",
            "cup_arm": "left",
            "minimum_lidar_clearance_m": 0.75,
        })
        self.assertFalse(route.ready)
        self.assertEqual(route.recovery_attempts, 15)
        self.assertEqual(route.recovery_wait_seconds, 2.0)
        self.assertEqual(route.base_settle_seconds, 3.0)

    def test_onsite_route_requires_bounded_narrow_door_calibration(self):
        payload = {
            "enabled": True,
            "bowl_arm": "right",
            "cup_arm": "left",
            "minimum_lidar_clearance_m": 0.75,
            "start_to_kitchen": [
                {"name": "align", "axis": "rotate", "distance": -10.0, "speed": 10.0},
                {"name": "door", "axis": "forward", "distance": 0.50,
                 "speed": 0.06, "narrow_passage": True},
            ],
            "dining_transition": [
                {"name": "dining", "axis": "strafe", "distance": 0.40, "speed": 0.10},
            ],
        }
        route = parse_onsite_bowl_cup_route(payload)
        self.assertTrue(route.ready)
        self.assertEqual(route.start_to_kitchen[0].distance, -10.0)
        self.assertTrue(route.start_to_kitchen[1].narrow_passage)
        self.assertEqual(route.recovery_attempts, 15)
        self.assertEqual(route.base_settle_seconds, 3.0)
        payload["start_to_kitchen"][1]["distance"] = None
        with self.assertRaises(ValueError):
            parse_onsite_bowl_cup_route(payload)

    def test_confirmed_relative_route_to_stage34(self):
        import json
        config_path = Path(__file__).parents[1] / "config" / "robot_io.json"
        raw = json.loads(config_path.read_text())["modes"]["real"]["onsite_bowl_cup_route"]
        expected = [
            ("forward", -1.60), ("forward", 1.60), ("strafe", 0.70),
            ("forward", 1.40), ("rotate", -90.0),
        ]
        self.assertEqual(
            [(m["axis"], m["distance"]) for m in raw["dining_transition"]], expected)
        self.assertTrue(raw["enabled"])
        self.assertTrue(raw["table_leg_tracking"]["enabled"])

    def test_autonomy_guard_allows_closed_loop_command_publisher(self):
        environment = {"AUTONOMOUS_ONLY": "1", "POLICY_MODE": "closed_loop"}
        self.assertEqual(environment_violations(environment), [])
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.py"
            policy.write_text("node.create_publisher(String, '/pedal/state', 10)\n")
            self.assertEqual(policy_subscription_violations(policy), [])

    def test_autonomy_guard_rejects_teleop(self):
        self.assertTrue(environment_violations({"WITH_GELLO_TELEOP": "true"}))
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.py"
            policy.write_text("node.create_subscription(String, '/pedal/state', cb, 10)\n")
            self.assertTrue(policy_subscription_violations(policy))

    def test_autonomy_guard_rejects_teleop_launch_but_not_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "entrypoint.sh"
            launch.write_text("# keyboard teleop is prohibited\nros2 launch pkg gello_teleop.launch.py\n")
            self.assertTrue(launch_violations(launch))
            launch.write_text("# keyboard teleop is prohibited\npython3 policy.py\n")
            self.assertEqual(launch_violations(launch), [])

    def test_lighting_quality_and_adaptive_gamma(self):
        import numpy as np
        black = np.zeros((20, 20, 3), dtype=np.uint8)
        self.assertFalse(lighting_quality(black)["usable"])
        dim = np.tile(np.arange(64, dtype=np.uint8), (20, 1))[:, :, None]
        dim = np.repeat(dim, 3, axis=2)
        self.assertTrue(lighting_quality(dim)["usable"])
        self.assertLess(adaptive_gamma(dim), 1.0)

    def test_safe_detour_or_fail_closed(self):
        obstacle = GroundObstacle("book", (2.0, 0.0), 0.25)
        route = plan_safe_path((0.0, 0.0), (4.0, 0.0), [obstacle])
        self.assertTrue(route.safe)
        self.assertGreater(len(route.waypoints), 1)
        self.assertTrue(segment_is_clear((0.0, 0.0), route.waypoints[0], [obstacle], 0.60))
        self.assertFalse(plan_safe_path((0.0, 0.0), (1.0, 0.0), None).safe)

    def test_official_ten_point_matrix(self):
        lighting = {profile: {stage: True for stage in ("stage1", "stage2", "stage3", "stage4")}
                    for profile in ("low_lux", "high_lux", "gradient_low", "gradient_high")}
        obstacles = {kind: {stage: {"task_success": True, "contact": False}
                            for stage in ("stage1", "stage2", "stage3", "stage4")}
                     for kind in ("cable", "book", "ball")}
        obstacles["ball"]["stage2"]["contact"] = True
        score = score_robustness(lighting, obstacles)
        self.assertEqual(score, {"illumination": 4.0, "avoidance": 5.5, "total": 9.5})
        self.assertEqual(score_complete_robustness(lighting, obstacles), score)

    def test_complete_evidence_is_required_for_a_score_claim(self):
        with self.assertRaises(EvidenceError):
            score_complete_robustness({}, {})

    def test_onsite_preflight_is_read_only_and_fails_on_missing_or_wrong_types(self):
        observed = parse_topic_list_t(
            "/isaac/left_joint_states [sensor_msgs/msg/JointState]\n"
            "/pedal/state [std_msgs/msg/String]\n"
        )
        issues = validate_topic_types(observed)
        self.assertIn("/isaac/right_joint_states", issues["missing"])
        self.assertEqual(issues["mismatched"], {})
        observed["/pedal/state"] = "geometry_msgs/msg/Twist"
        self.assertIn("/pedal/state", validate_topic_types(observed)["mismatched"])

    def test_onsite_preflight_fixture_is_complete(self):
        fixture = Path(__file__).parent / "fixtures" / "task3_sim_bridge_topics.json"
        import json
        observed = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(validate_topic_types(observed), {"missing": {}, "mismatched": {}})

    def test_real_onsite_preflight_uses_only_documented_topics(self):
        self.assertNotIn("/left/joint_commands", EXPECTED_REAL_PLATFORM_TYPES)
        self.assertEqual(
            EXPECTED_REAL_PLATFORM_TYPES["/swerve_drive_controller/cmd_vel"],
            "geometry_msgs/msg/TwistStamped")
        self.assertEqual(
            EXPECTED_REAL_PLATFORM_TYPES["/lidar_front/scan"],
            "sensor_msgs/msg/LaserScan")


if __name__ == "__main__":
    unittest.main()
