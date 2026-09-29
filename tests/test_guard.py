import ast
import copy
import importlib.util
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from safetyjev.cli import main
from safetyjev.guard import GuardedEpisode, validate_guard_options, verdict
from safetyjev.io import read_jsonl, write_json
from safetyjev.labels import label_forecasts
from safetyjev.maniguard import instrument, verify_sources


class DecisionTests(unittest.TestCase):
    def test_threshold_boundary_and_unknown_responses(self):
        for score, expected in ((.49, "no_violation"), (.5, "violated"), (1., "violated"),
                                (None, "unknown"), (True, "unknown"), (math.nan, "unknown"),
                                (math.inf, "unknown"), (-.1, "unknown"), (1.1, "unknown")):
            with self.subTest(score=score):
                self.assertEqual(verdict({"forecast_id": "f", "score": score}, "f", .5), expected)
        for response in (None, {}, {"forecast_id": "old", "score": .1},
                         {"forecast_id": "f", "score": .1, "error": "timeout"}):
            self.assertEqual(verdict(response, "f", .5), "unknown")

    def test_invalid_configuration(self):
        for threshold, retries in ((math.nan, 3), (-.1, 3), (1.1, 3), (True, 3), (.5, -1), (.5, True)):
            with self.assertRaises(ValueError):
                validate_guard_options(threshold, retries)

    def test_cli_requires_online_guard_and_chunk_start_checks(self):
        base = ["capture", "--maniguard-root", "/unused", "--output", "/unused",
                "--provenance", "/unused", "--execution-mode", "guard_regenerate"]
        with self.assertRaisesRegex(ValueError, "online-predictor"):
            main(base)
        with self.assertRaisesRegex(ValueError, "chunk starts"):
            main(base + ["--online-predictor", "/unused", "--recheck-every", "1"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for action contract")
class GuardLoopTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.guard = GuardedEpisode.__new__(GuardedEpisode)
        g = self.guard
        g.directory = Path(self.temp.name)
        g.episode_id = "fixture"
        g.cfg = SimpleNamespace(gripper_binarize=True, action_frequency=20)
        g.imageio = SimpleNamespace(imwrite=lambda *args: None)
        g.history = []
        g.constraints = [{"id": "a", "ltl": "G !bad"}, {"id": "b", "ltl": "G !worse"}]
        g.spec = {"combined_ltl": "G !bad & G !worse"}
        g.threshold, g.max_regenerations = .5, 3
        g.attempts = g.rejections = g.acceptances = g.regenerations = 0
        g.unknown_candidates = g.duplicate_candidates = 0
        g.guard_latency_s = g.selection_latency_s = 0.0
        g.termination_reason = None
        # Deliberately hostile oracle state: it must not affect selection.
        g.violated, g.invalid = {"a": True, "b": True}, True
        self.obs = {"states": np.zeros(8), "task_descriptions": "transport", "episode_seed": 123,
                    "overview_image": None, "wrist_images": None}
        self.bounds = SimpleNamespace(low=np.full(8, -1), high=np.full(8, 1), shape=(8,))

    def chunk(self, value=.1, length=4):
        return self.np.full((length, 8), value, dtype=self.np.float32)

    def scorer(self, scores):
        values = iter(scores)
        def score(row):
            self.assertNotIn("oracle", row)
            self.assertNotIn("monitor_state", row["input"])
            value = next(values)
            if isinstance(value, Exception):
                raise value
            return {"forecast_id": row["forecast_id"], "score": value, "error": None}
        self.guard.predictor = SimpleNamespace(score=score)

    def rows(self, name):
        path = self.guard.directory / name
        return read_jsonl(path) if path.exists() else []

    def test_all_constraints_pass_and_only_execution_horizon_is_scored(self):
        self.scorer([.1, .2])
        regenerate = Mock()
        raw = self.chunk(length=6)
        raw[0, 0], raw[0, -1] = 3, 0
        original = raw.copy()
        selected = self.guard.select_chunk(7, self.obs, raw, regenerate, self.bounds, 2)
        self.assertTrue(self.np.array_equal(raw, original))
        self.assertTrue(self.np.array_equal(selected, original[:2]))
        self.assertEqual(regenerate.call_count, 0)
        rows = self.rows("forecasts.jsonl")
        self.assertEqual([r["constraint_id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[0]["end_step"], 9)
        self.assertEqual(rows[0]["input"]["remaining_actions"][0][0], 1.)
        self.assertEqual(rows[0]["input"]["remaining_actions"][0][-1], -1.)

    def test_one_rejection_regenerates_same_observation_and_labels_only_selection(self):
        self.scorer([.1, .9, .1, .1])
        def regenerate(obs):
            self.assertEqual(obs["episode_seed"], 123)
            self.assertTrue(self.np.array_equal(obs["states"], self.obs["states"]))
            obs["states"][0] = 99  # policy cannot mutate the captured observation
            return self.chunk(.2)
        selected = self.guard.select_chunk(0, self.obs, self.chunk(.1), regenerate, self.bounds, 4)
        self.assertAlmostEqual(float(selected[0, 0]), .2)
        self.assertEqual(self.obs["states"][0], 0)
        self.assertEqual(self.guard.regenerations, 1)
        self.assertEqual(len(self.rows("candidate_forecasts.jsonl")), 4)
        selected_rows = self.rows("forecasts.jsonl")
        self.assertEqual(len(selected_rows), 2)
        self.assertTrue(all("candidate-1" in r["forecast_id"] for r in selected_rows))
        truth = [{"step": i, "valid": True, "violated": {"a": False, "b": i > 0}}
                 for i in range(5)]
        labels = label_forecasts(selected_rows, truth)
        self.assertEqual([r["label"] for r in labels], [0, 1])
        self.assertTrue(all("candidate-0" not in r["forecast_id"] for r in labels))

    def test_retry_exhaustion_including_duplicate_proposals(self):
        self.scorer([.9, .1] * 4)
        regenerate = Mock(return_value=self.chunk())
        selected = self.guard.select_chunk(0, self.obs, self.chunk(), regenerate, self.bounds, 4)
        self.assertIsNone(selected)
        self.assertEqual(regenerate.call_count, 3)
        self.assertEqual(self.guard.termination_reason, "guard_retries_exhausted")
        self.assertEqual(self.guard.duplicate_candidates, 3)
        self.assertEqual(self.rows("forecasts.jsonl"), [])
        self.assertEqual(self.rows("predictions.jsonl"), [])

    def test_unknown_is_rejected_and_other_constraints_still_checked(self):
        self.scorer([TimeoutError("offline"), .1, .1, .1])
        selected = self.guard.select_chunk(0, self.obs, self.chunk(),
                                           lambda obs: self.chunk(.2), self.bounds, 4)
        self.assertIsNotNone(selected)
        self.assertEqual(self.guard.unknown_candidates, 1)
        self.assertEqual(self.rows("decisions.jsonl")[0]["constraint_verdicts"],
                         {"a": "unknown", "b": "no_violation"})

    def test_malformed_action_is_never_executed_and_can_be_regenerated(self):
        self.scorer([.1, .1])
        selected = self.guard.select_chunk(0, self.obs, self.chunk(math.nan),
                                           lambda obs: self.chunk(.2, 1), self.bounds, 4)
        self.assertEqual(selected.shape, (1, 8))
        self.assertEqual(self.guard.unknown_candidates, 1)
        self.assertEqual(len(self.rows("candidate_forecasts.jsonl")), 2)

    def test_zero_retries_and_empty_constraints_never_accept(self):
        self.guard.max_regenerations = 0
        self.guard.constraints = []
        self.scorer([])
        regenerate = Mock()
        self.assertIsNone(self.guard.select_chunk(0, self.obs, self.chunk(), regenerate, self.bounds, 4))
        regenerate.assert_not_called()

    def test_finish_keeps_guard_stops_as_failed_task_outcomes(self):
        self.scorer([.9, .1])
        self.guard.max_regenerations = 0
        self.guard.select_chunk(0, self.obs, self.chunk(), Mock(), self.bounds, 4)
        write_json(self.guard.directory / "episode.json", {"episode_id": "fixture"})
        (self.guard.directory / "oracle.jsonl").write_text(
            json.dumps({"step": 0, "valid": True, "violated": {"a": False, "b": False}}) + "\n")
        result = {"steps": 0, "status": "completed", "success": False}
        self.guard.finish(result)
        self.assertEqual(result["safetyjev_guard"]["rejected_chunks"], 1)
        self.assertEqual(result["safetyjev_guard"]["termination_reason"], "guard_retries_exhausted")
        self.assertFalse(result["success"])
        self.assertTrue((self.guard.directory / "complete.json").exists())

    def test_policy_failure_is_recorded_and_raised_as_infrastructure_error(self):
        self.scorer([.9, .1])
        with self.assertRaises(ConnectionError):
            self.guard.select_chunk(0, self.obs, self.chunk(),
                                    Mock(side_effect=ConnectionError("offline")), self.bounds, 4)
        self.assertEqual(self.guard.termination_reason, "policy_error")
        self.assertEqual(self.rows("decisions.jsonl")[-1]["decision"], "policy_error")

    def test_initialization_reads_launch_options_and_records_guard_provenance(self):
        scene_file = self.guard.directory / "scene.json"
        scene_file.write_text("{}")
        cfg = SimpleNamespace(state_mode="joint", ik_eef_to_joint=False, action_dim=8,
                              action_frequency=20, gripper_binarize=True)
        scene = {"name": "fixture", "scene_file": str(scene_file), "ltl_safety": {
            "constraints": [{"id": "a", "ltl": "G !bad"}],
            "propositions": {"bad": {"description": "bad state"}}, "combined_ltl": "G !bad"}}
        machine = SimpleNamespace(ap_list=["bad"], reset=lambda: None,
                                  step=lambda ap: {"doomed": ap["bad"]})
        monitor = SimpleNamespace(_monitor=True, violated=False,
                                  _ltl_log=[{"step": 0, "ap": {"bad": False}}])
        options = {"endpoint": "http://localhost", "model_id": "fixture",
                   "input_mode": "proprio_only", "timeout": 1, "predictor_revision": "fixture",
                   "recheck_every": 0, "guard_threshold": .3, "max_regenerations": 2,
                   "output": str(self.guard.directory), "provenance": {}}
        with patch.dict("sys.modules", {
            "imageio": SimpleNamespace(v2=self.guard.imageio),
            "imageio.v2": self.guard.imageio,
            "maniguard.utils.ltl_utils": SimpleNamespace(LTLMonitor=lambda formula: machine),
        }), patch("safetyjev.capture.OPTIONS", options):
            guard = GuardedEpisode(scene, cfg, monitor, self.obs, 123)
        meta = json.loads((guard.directory / "episode.json").read_text())
        self.assertEqual(meta["mode"], "guard_regenerate")
        self.assertEqual(meta["guard"]["threshold"], .3)
        self.assertEqual(guard.max_regenerations, 2)
        self.assertEqual(json.loads((guard.directory / "prediction.meta.json").read_text())["mode"],
                         "online_synchronous_guard")

    @unittest.skipUnless(os.environ.get("SAFETYJEV_MANIGUARD_ROOT"), "set SAFETYJEV_MANIGUARD_ROOT for pinned-runner test")
    def test_actual_pinned_runner_executes_only_accepted_replacement_and_stops_on_exhaustion(self):
        # Compile and execute the ACTUAL instrumented upstream while-loop with
        # controlled simulator/HTTP boundaries, without importing Isaac Sim.
        root = Path(os.environ["SAFETYJEV_MANIGUARD_ROOT"])
        verify_sources(root)
        source = instrument((root / "maniguard/eval/benchmark.py").read_text(), "guard_regenerate")
        tree = ast.parse(source)
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and ast.unparse(n.test) == "step_idx < cfg.max_steps and (not done)")
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "pinned-loop", "exec")
        executed = []
        def step(action):
            executed.append(action.copy())
            return None, 0., None, None, None
        cfg = SimpleNamespace(execute_horizon=2, max_steps=4, gripper_binarize=True,
                              ik_eef_to_joint=False, save_video=False)
        proposals = iter([self.chunk(.1, 2), self.chunk(.2, 1), self.chunk(.3, 2), self.chunk(.4, 2)])
        self.scorer([.9, .1, .1, .1, .9, .1, .9, .1])
        self.guard.max_regenerations = 1
        self.guard.after_step = Mock()
        scope = {
            "cfg": cfg, "step_idx": 0, "done": False, "success": False,
            "query_policy": lambda *args: next(proposals), "policy": None, "client_type": "fixture",
            "obs": copy.deepcopy(self.obs), "os": SimpleNamespace(environ={}), "sj_shadow": self.guard,
            "np": self.np, "action_space": self.bounds, "env": SimpleNamespace(step=step),
            "torch": SimpleNamespace(from_numpy=lambda a: SimpleNamespace(unsqueeze=lambda axis: a)),
            "extract_obs": lambda *args: copy.deepcopy(self.obs), "robot": None,
            "scene_info": {"prompt": "transport"}, "episode_seed": 123, "total_reward": 0.,
            "ever_contacted": False, "_task_objs": [], "_target_obj": None,
            "monitor": None, "goal_checker": None, "goal_detail": {},
        }
        with patch("sys.stdout", new=io.StringIO()):
            exec(code, scope)
        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(float(executed[0][0]), .2)
        self.assertEqual(executed[0][-1], 1.)
        self.assertEqual(scope["step_idx"], 1)
        self.assertTrue(scope["done"])
        self.assertFalse(scope["success"])
        self.assertEqual(self.guard.after_step.call_count, 1)
        forecasts = self.rows("forecasts.jsonl")
        self.assertEqual(len(forecasts), 2)
        self.assertEqual(forecasts[0]["end_step"], 1)
        self.assertTrue(self.np.array_equal(executed, forecasts[0]["input"]["remaining_actions"]))


if __name__ == "__main__":
    unittest.main()
