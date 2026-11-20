import json
import math
import tempfile
import unittest
from pathlib import Path

from zeus.brain import ZeusBrain
from zeus.memory import ExperienceMemory
from zeus.simulator import GridWorld


class ZeusBrainTests(unittest.TestCase):
    def test_selects_highest_score_and_explains(self):
        result = ZeusBrain().decide({"a": {"move_toward_goal": 1},
                                     "b": {"move_toward_goal": -1}})
        self.assertEqual(result.action, "a")
        self.assertGreater(result.confidence, 0.5)
        self.assertIn("feature score", result.reason)

    def test_empty_candidates_return_none(self):
        self.assertIsNone(ZeusBrain().decide({}))

    def test_one_candidate_is_not_claimed_as_certain(self):
        self.assertEqual(ZeusBrain().decide({"only": {}}).confidence, 0.5)

    def test_ties_preserve_input_order(self):
        self.assertEqual(ZeusBrain().decide({"first": {}, "second": {}}).action, "first")

    def test_learning_is_bounded(self):
        brain = ZeusBrain(learning_rate=1.0)
        brain.learn("east", 1, {"move_toward_goal": 100})
        self.assertEqual(brain.weights["move_toward_goal"], 2.0)
        self.assertEqual(brain.status()["feedback_count"], 1)

    def test_learning_clamps_reward(self):
        brain = ZeusBrain()
        brain.learn("east", 100, {"x": 1})
        self.assertAlmostEqual(brain.weights["x"], 0.1)

    def test_invalid_learning_input_does_not_partially_update(self):
        brain = ZeusBrain()
        before = brain.status()
        with self.assertRaises(ValueError):
            brain.learn("east", 1, {"valid": 1, "bad": math.nan})
        self.assertEqual(brain.status(), before)

    def test_rejects_non_finite_decision_features(self):
        with self.assertRaises(ValueError):
            ZeusBrain().decide({"a": {"x": float("inf")}})

    def test_rejects_score_overflow(self):
        brain = ZeusBrain()
        brain.weights["x"] = 2.0
        brain.weights["y"] = 2.0
        with self.assertRaises(ValueError):
            brain.decide({"a": {"x": 1e308, "y": 1e308}})

    def test_rejects_invalid_weight_bounds(self):
        with self.assertRaises(ValueError):
            ZeusBrain(min_weight=2, max_weight=1)

    def test_similar_positive_experience_influences_decision(self):
        brain = ZeusBrain(experience_scale=1.0)
        options = {"east": {"feature": 1.0}, "west": {"feature": 1.0}}
        experiences = [{"action": "east", "features": {"feature": 1.0}, "reward": 1.0}]
        self.assertEqual(brain.decide(options, experiences).action, "east")

    def test_unrelated_experience_does_not_dilute_relevant_memory(self):
        brain = ZeusBrain(experience_scale=1.0)
        options = {"east": {"feature": 1.0}, "west": {"feature": 1.0}}
        experiences = [
            {"action": "east", "features": {"feature": 1.0}, "reward": 1.0},
            {"action": "east", "features": {"feature": 100.0}, "reward": -1.0},
        ]
        self.assertEqual(brain.decide(options, experiences).action, "east")


class MemoryTests(unittest.TestCase):
    def test_memory_round_trip_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ExperienceMemory(str(Path(tmp) / "memory.jsonl"))
            memory.record({"x": 1}, "east", 0.5, "safe", {"toward": 1})
            self.assertEqual(len(memory.all()), 1)
            self.assertAlmostEqual(memory.summary()["average_reward"], 0.5)

    def test_memory_clamps_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ExperienceMemory(str(Path(tmp) / "memory.jsonl"))
            memory.record({}, "east", 5, "safe")
            self.assertEqual(memory.all()[0]["reward"], 1.0)

    def test_memory_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ExperienceMemory(str(Path(tmp) / "memory.jsonl"))
            with self.assertRaises(ValueError):
                memory.record({}, "east", float("nan"), "safe")
            self.assertEqual(memory.all(), [])

    def test_memory_reports_corrupt_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            path.write_text('{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                ExperienceMemory(str(path)).all()

    def test_memory_rejects_non_json_state_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ExperienceMemory(str(Path(tmp) / "memory.jsonl"))
            with self.assertRaises(ValueError):
                memory.record({"bad": object()}, "east", 0.2, "safe")
            self.assertEqual(memory.all(), [])


class SimulatorTests(unittest.TestCase):
    def test_world_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            GridWorld().step("teleport")

    def test_world_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            GridWorld(width=0)

    def test_world_stops_after_goal(self):
        world = GridWorld(width=2, height=2, position=(0, 0), goal=(1, 0), hazard=(1, 1))
        reward, outcome = world.step("east")
        self.assertEqual((reward, outcome), (1.0, "goal reached"))
        self.assertTrue(world.terminal)
        self.assertEqual(world.options(), {})
        with self.assertRaises(ValueError):
            world.step("west")

    def test_world_stops_after_hazard(self):
        world = GridWorld(width=2, height=2, position=(0, 0), goal=(1, 0), hazard=(0, 1))
        reward, outcome = world.step("south")
        self.assertEqual((reward, outcome), (-1.0, "hazard reached"))
        with self.assertRaises(ValueError):
            world.step("north")


if __name__ == "__main__":
    unittest.main()
