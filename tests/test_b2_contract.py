import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from b2_contract import (
    ContractError,
    DetectionObservation,
    load_contract,
    normalize_object_name,
    parse_detection_name,
    quaternion_transform_point,
    required_topics,
    stale_keys,
    transform_point,
)
from task3_autonomous import seat_pose_from_observation
from yolo_detector import TASK3_CLASSES, missing_task3_classes
from vision_callback import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_both_modes_have_unique_required_topics(self):
        for mode in ("sim", "real"):
            contract = load_contract(ROOT / "config" / "robot_io.json", mode)
            topics = required_topics(contract)
            # Real base entry must not be blocked by a 3-D perception publisher:
            # the released interface advertises RGB but no depth/CameraInfo.
            self.assertEqual(len(topics), 8 if mode == "sim" else 9)
            self.assertEqual(len(set(topics)), len(topics))
            self.assertEqual(len(contract["nav_targets"]["kitchen"]), 3)
            self.assertGreater(contract["limits"]["force_stop_threshold_n"], 0)
            self.assertEqual(contract["static_required_topic_keys"], [])
            self.assertNotEqual(
                contract["gripper_command_open"], contract["gripper_command_closed"])

    def test_official_real_topic_mapping_is_confirmed(self):
        contract = load_contract(ROOT / "config" / "robot_io.json", "real")
        topics = contract["topics"]
        self.assertTrue(contract["confirmed"])
        self.assertFalse(contract["vision_3d_available"])
        self.assertTrue(contract["obstacle_safety_required"])
        self.assertEqual(contract["base_command_type"], "twist_stamped")
        self.assertEqual(topics["base_cmd"], "/swerve_drive_controller/cmd_vel")
        self.assertEqual(topics["odom"], "/swerve_drive_controller/odom")
        self.assertEqual(topics["front_lidar"], "/lidar_front/scan")
        self.assertEqual(topics["rear_lidar"], "/lidar_rear/scan")
        self.assertEqual(topics["left_arm_cmd"], "/left/gello/joint_states")
        self.assertEqual(topics["right_arm_cmd"], "/right/gello/joint_states")
        self.assertEqual(topics["head_rgb"], "/head_camera/zed_node/rgb/color/rect/image")
        self.assertNotIn("head_depth", topics)

    def test_rgbd_contract_is_required_only_when_declared(self):
        sim = load_contract(ROOT / "config" / "robot_io.json", "sim")
        real = load_contract(ROOT / "config" / "robot_io.json", "real")
        self.assertTrue(sim["vision_3d_available"])
        self.assertIn("head_depth", sim["topics"])
        self.assertFalse(real["vision_3d_available"])
        self.assertNotIn("head_camera_info", real["topics"])
        self.assertNotIn("vision_objects", real["required_topic_keys"])

    def test_sim_commands_match_the_official_shared_bridge_contract(self):
        sim = load_contract(ROOT / "config" / "robot_io.json", "sim")
        topics = sim["topics"]
        self.assertEqual(topics["left_arm_cmd"], "/isaac/left_joint_commands")
        self.assertEqual(topics["right_arm_cmd"], "/isaac/right_joint_commands")
        self.assertEqual(topics["left_gripper_cmd"], "/isaac/left_robotiq_joint_commands")
        self.assertEqual(topics["right_gripper_cmd"], "/isaac/right_robotiq_joint_commands")
        self.assertEqual(topics["pedal_cmd"], "/pedal/state")
        self.assertEqual(topics["base_cmd"], "/pedal/state")
        self.assertNotIn("target_seat_pose", topics)

    def test_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema_version": 99, "modes": {}}))
            with self.assertRaises(ContractError):
                load_contract(path, "sim")

    def test_detection_name_and_freshness(self):
        self.assertEqual(parse_detection_name("bowl:0.87"), ("bowl", 0.87))
        self.assertEqual(normalize_object_name("bowl2"), "bowl")
        self.assertEqual(normalize_object_name("ikea_knock_box"), "recycling_bin")
        self.assertEqual(normalize_object_name("mouse"), None)
        now = time.monotonic()
        observation = DetectionObservation((1, 2, 3), 0.9, "world", 1.0, now)
        self.assertTrue(observation.is_fresh(now + 0.5, 1.0))
        self.assertFalse(observation.is_fresh(now + 1.5, 1.0))

    def test_visual_seat_target_uses_observation_not_static_topic(self):
        observation = DetectionObservation(
            position=(2.0, 3.0, 0.8), confidence=0.9, frame_id="world",
            source_stamp=1.0, received_at=time.monotonic())
        pose = seat_pose_from_observation(observation, (1.0, 3.0))
        self.assertEqual(pose[:3], (2.0, 3.0, 0.8))
        self.assertAlmostEqual(pose[3], 0.0, places=7)
        self.assertIsNone(seat_pose_from_observation(None, (1.0, 3.0)))
        self.assertIsNone(seat_pose_from_observation(observation, (2.0, 3.0)))

    def test_yolo_contract_uses_one_task3_label_set(self):
        expected = list(OBJECT_CLASSES.values())
        self.assertEqual(TASK3_CLASSES, expected)
        self.assertEqual(missing_task3_classes(expected), [])
        self.assertEqual(missing_task3_classes(["plate", "cup"]), expected[2:])

    def test_stale_keys_are_fail_closed(self):
        self.assertEqual(stale_keys(("joint", "vision"), {"joint": 10.0}, 10.5, 1.0), ("vision",))
        self.assertEqual(stale_keys(("joint",), {"joint": 8.0}, 10.5, 1.0), ("joint",))
        self.assertEqual(
            stale_keys(("seat",), {"seat": 1.0}, 100.0, 1.0, static_keys=("seat",)),
            (),
        )
        self.assertEqual(
            stale_keys(("seat",), {}, 100.0, 1.0, static_keys=("seat",)),
            ("seat",),
        )
        self.assertEqual(
            stale_keys(("vision",), {"vision": 8.0}, 10.0, 1.0,
                       max_age_by_key={"vision": 3.0}),
            (),
        )

    def test_transforms(self):
        matrix = ((1, 0, 0, 1), (0, 1, 0, 2), (0, 0, 1, 3), (0, 0, 0, 1))
        self.assertEqual(transform_point(matrix, (1, 1, 1)), (2.0, 3.0, 4.0))
        # 90 degrees about Z plus translation.
        q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        transformed = quaternion_transform_point((1, 0, 0), (1, 2, 3), q)
        self.assertAlmostEqual(transformed[0], 1.0, places=7)
        self.assertAlmostEqual(transformed[1], 3.0, places=7)
        self.assertAlmostEqual(transformed[2], 3.0, places=7)


if __name__ == "__main__":
    unittest.main()
