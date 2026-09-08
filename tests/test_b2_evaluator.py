import unittest

from b2_evaluator import evaluate_episode


SPEC = {
    "dining_bounds": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
    "sink_bounds": {"x": [2, 3], "y": [2, 3], "z": [0, 2]},
    "feeding": {"required_hold_seconds": 3.0, "feed_zone_tolerance_m": 0.1,
                "max_sample_gap_seconds": 0.5, "max_motion_step_m": 1.5},
    "recovery": {"container_center_xyz": [5, 5, 1], "container_radius_m": 0.2},
}


def feed_samples(beans=4, count=17):
    return [{"timestamp": i * 0.2, "spoon_xyz": [0, 0, 1],
             "feed_target_xyz": [0, 0, 1], "beans_on_spoon": beans}
            for i in range(count)]


class EvaluatorTests(unittest.TestCase):
    def episode(self):
        return {
            "stage1_final_objects": {name: [0, 0, 1] for name in ("plate2", "cup", "bowl2", "spoon2")},
            "feed_samples": feed_samples(),
            "stage3": {"beans_before": 10, "bean_positions_after": [[5, 5, 1]] * 10},
            "stage4_final_objects": {name: [2.5, 2.5, 1] for name in ("plate2", "cup", "bowl2", "spoon2")},
            "score": 999,
        }

    def test_full_observed_episode_scores_16(self):
        result = evaluate_episode(self.episode(), SPEC)
        self.assertEqual(result["total_score"], 16)
        self.assertEqual(result["max_score"], 16)

    def test_empty_spoon_fails_even_with_long_hold(self):
        episode = self.episode()
        episode["feed_samples"] = feed_samples(beans=0, count=30)
        self.assertEqual(evaluate_episode(episode, SPEC)["stages"]["stage2"]["score"], 0)

    def test_completed_feed_is_retained_after_later_invalid_sample(self):
        episode = self.episode()
        samples = feed_samples(beans=3, count=17)
        samples.append({"timestamp": 3.4, "spoon_xyz": [1, 1, 1],
                        "feed_target_xyz": [0, 0, 1], "beans_on_spoon": 0})
        episode["feed_samples"] = samples
        stage2 = evaluate_episode(episode, SPEC)["stages"]["stage2"]
        self.assertEqual(stage2["score"], 3)
        self.assertTrue(stage2["completed"])

    def test_controller_score_field_is_ignored(self):
        episode = self.episode()
        episode["stage1_final_objects"] = {}
        episode["score"] = 16
        self.assertEqual(evaluate_episode(episode, SPEC)["stages"]["stage1"]["score"], 0)

    def test_unknown_recovery_scores_zero(self):
        episode = self.episode()
        episode["stage3"] = {"beans_before": 0, "bean_positions_after": []}
        stage3 = evaluate_episode(episode, SPEC)["stages"]["stage3"]
        self.assertEqual(stage3["score"], 0)
        self.assertFalse(stage3["observable"])


if __name__ == "__main__":
    unittest.main()
