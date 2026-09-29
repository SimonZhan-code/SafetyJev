import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from safetyjev.capture import ShadowEpisode
from safetyjev.io import read_jsonl
from safetyjev.maniguard import instrument


class CaptureTests(unittest.TestCase):
    def test_monitor_gap_invalidates_following_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            shadow = ShadowEpisode.__new__(ShadowEpisode)
            shadow.directory = Path(temp)
            shadow.invalid, shadow.last_step = False, -1
            shadow.machines = {"a": SimpleNamespace(ap_list=["bad"], step=lambda ap: {"doomed": ap["bad"]})}
            shadow.violated = {"a": False}
            shadow.history, shadow.last_state, shadow.last_observation_step = [], [0], 0
            obs = {"states": SimpleNamespace(tolist=lambda: [0])}
            monitor = SimpleNamespace(_monitor=True, _ltl_log=[{"step": 0, "ap": {"bad": False}}], violated=False)
            shadow.after_step(0, monitor, obs)
            # Step 1's upstream monitor failed and did not append a sample.
            shadow.after_step(1, monitor, obs)
            monitor._ltl_log.append({"step": 2, "ap": {"bad": True}})
            monitor.violated = True
            shadow.after_step(2, monitor, obs)
            self.assertEqual([r["valid"] for r in read_jsonl(Path(temp) / "oracle.jsonl")], [True, False, False])

    def test_missing_ap_invalidates_even_when_combined_is_false(self):
        with tempfile.TemporaryDirectory() as temp:
            shadow = ShadowEpisode.__new__(ShadowEpisode)
            shadow.directory = Path(temp)
            shadow.invalid, shadow.last_step = False, -1
            shadow.machines = {"a": SimpleNamespace(ap_list=["required"])}
            shadow.violated = {"a": False}
            monitor = SimpleNamespace(_monitor=True, _ltl_log=[{"step": 0, "ap": {}}], violated=False)
            shadow.after_step(0, monitor, {})
            self.assertTrue(shadow.invalid)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional NumPy for simulator action contract")
    def test_capture_matches_joint_commands_without_mutating_chunk(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            shadow = ShadowEpisode.__new__(ShadowEpisode)
            shadow.directory = Path(temp)
            shadow.episode_id = "fixture"
            shadow.cfg = SimpleNamespace(gripper_binarize=True, action_frequency=20)
            shadow.imageio = SimpleNamespace(imwrite=lambda *args: None)
            shadow.history = []
            shadow.constraints = [{"id": "a", "ltl": "G !bad"}]
            shadow.spec = {"combined_ltl": "G !bad"}
            shadow.predictor = None
            chunk = np.zeros((4, 8), dtype=np.float32)
            chunk[:, -1] = [0, .2, -.3, .001]
            chunk[1, 0] = 3
            original = chunk.copy()
            obs = {"states": np.zeros(8), "task_descriptions": "transport",
                   "overview_image": None, "wrist_images": None}
            bounds = SimpleNamespace(low=np.full(8, -1), high=np.full(8, 1))
            with patch("safetyjev.capture.OPTIONS", {"recheck_every": 1}):
                shadow.before_action(11, obs, chunk, 1, 3, bounds)
            rows = read_jsonl(Path(temp) / "forecasts.jsonl")
            self.assertTrue(np.array_equal(chunk, original))
            self.assertEqual(rows[0]["end_step"], 13)
            self.assertEqual(rows[0]["input"]["remaining_actions"][0][0], 1)
            self.assertEqual([r[-1] for r in rows[0]["input"]["remaining_actions"]], [1, -1])
            self.assertEqual([r["constraint_id"] for r in rows], ["a", "__all__"])

    def test_changed_source_is_refused(self):
        with self.assertRaises(ValueError):
            instrument("print('unreviewed runner')")


if __name__ == "__main__":
    unittest.main()
