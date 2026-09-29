import ast
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from safetyjev.cli import main
from safetyjev.guard import GuardedEpisode
from safetyjev.capture import ShadowEpisode
from safetyjev.maniguard import instrument, verify_sources
from safetyjev.planner import OpenRouterPlanner, PlannerError
from safetyjev.predictors import OpenJevHTTP


def context():
    return {"original_task": "Move the closed jar", "current_instruction": "Move the closed jar",
            "start_step": 8, "robot_state": [0.] * 8, "remaining_actions": [[.1] * 8],
            "action_convention": "absolute joint", "action_frequency_hz": 20,
            "constraints": [{"id": "lid", "ltl": "G !bad", "description": "Close before lift"}],
            "guard_feedback": [{"constraint_id": "lid", "score": .8, "verdict": "violated"}],
            "executed_history": [{"step": 8, "robot_state": [0.] * 8,
                                  "executed_action": [.1] * 8, "policy_instruction": "Approach jar"}],
            "candidate_history": [{"candidate_id": "e:8:0", "start_step": 8,
                                   "policy_instruction": "Lift jar", "decision": "reject",
                                   "guard_feedback": [{"constraint_id": "lid", "score": .8,
                                                       "verdict": "violated"}]}],
            "images": {"overview_image": "overview.png", "wrist_images": "wrist.png"}}


def response(instruction="Close the jar lid before lifting.", finish="stop", content=None):
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": finish,
                                               "message": {"content": content if content is not None else
                                                            json.dumps({"vla_instruction": instruction})}}],
                                 "usage": {"prompt_tokens": 500, "completion_tokens": 20, "cost": .001},
                                 "model": "provider/test"}).encode())


class PlannerAPITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "TEST_SECRET_KEY", "OPENROUTER_MODEL": "provider/test"})
        env.start()
        self.addCleanup(env.stop)
        self.planner = OpenRouterPlanner({"include_images": False, "history_limit": 1})

    def test_request_allowlist_and_bounded_history(self):
        c = context()
        c["oracle"] = c["monitor_state"] = "FORBIDDEN_ORACLE"
        for key in ("constraints", "guard_feedback", "candidate_history", "executed_history"):
            c[key][0]["oracle"] = "FORBIDDEN_ORACLE"
        c["executed_history"] *= 4
        body, audit = self.planner.request_body(c, "/unused")
        serialized = json.dumps(body)
        self.assertNotIn("FORBIDDEN_ORACLE", serialized)
        self.assertNotIn("TEST_SECRET_KEY", serialized)
        self.assertNotIn("TEST_SECRET_KEY", json.dumps(self.planner.metadata()))
        self.assertEqual(len(audit["context"]["executed_history"]), 1)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["provider"]["require_parameters"])

    def test_http_request_response_and_no_secret_in_artifact(self):
        with patch("safetyjev.planner.urlopen", return_value=response()) as send:
            plan = self.planner.plan(context(), "/unused")
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer TEST_SECRET_KEY")
        self.assertEqual(plan["vla_instruction"], "Close the jar lid before lifting.")
        self.assertEqual(plan["usage"]["cost"], .001)
        self.assertEqual(plan["response_model"], "provider/test")
        self.assertNotIn("TEST_SECRET_KEY", json.dumps(plan))

    def test_images_are_current_pngs_and_paths_cannot_escape_episode(self):
        self.planner.include_images = True
        with tempfile.TemporaryDirectory() as temp:
            for name in ("overview.png", "wrist.png"):
                (Path(temp) / name).write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            body, audit = self.planner.request_body(context(), temp)
            parts = body["messages"][1]["content"]
            self.assertEqual(sum(p["type"] == "image_url" for p in parts), 2)
            self.assertEqual(len(audit["images"]), 2)
            self.assertNotIn("base64", json.dumps(audit))
            c = context()
            c["images"]["overview_image"] = "../outside.png"
            with self.assertRaises(PlannerError):
                self.planner.request_body(c, temp)

    def test_malformed_truncated_empty_and_extra_field_outputs_rejected(self):
        cases = [response(finish="length"), response(instruction=" "), response(instruction="x" * 513),
                 response(content="not json"), response(content='{"vla_instruction":"ok","disable_guard":true}'),
                 response(instruction="line\nbreak"), response(instruction="TEST_SECRET_KEY")]
        for result in cases:
            with self.subTest(result=result), patch("safetyjev.planner.urlopen", return_value=result):
                with self.assertRaises(PlannerError):
                    self.planner.plan(context(), "/unused")

    def test_service_errors_do_not_log_credential_or_server_body(self):
        for error in (TimeoutError("TEST_SECRET_KEY"),
                      HTTPError("url", 401, "TEST_SECRET_KEY", {}, io.BytesIO(b"TEST_SECRET_KEY"))):
            with patch("safetyjev.planner.urlopen", side_effect=error):
                with self.assertRaises(PlannerError) as raised:
                    self.planner.plan(context(), "/unused")
                self.assertNotIn("TEST_SECRET_KEY", str(raised.exception))

    def test_configuration_requires_credentials_and_bounded_settings(self):
        for config in ({"api_key": "forbidden"}, {"history_limit": 0}, {"max_calls_per_episode": -1},
                       {"timeout": float("nan")}, {"include_images": "true"}, {"max_tokens": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                OpenRouterPlanner(config)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            OpenRouterPlanner({"model": "provider/test"})

    def test_planner_cannot_be_enabled_in_shadow_mode(self):
        with self.assertRaisesRegex(ValueError, "guard_regenerate"):
            main(["capture", "--maniguard-root", "/unused", "--output", "/unused",
                  "--provenance", "/unused", "--planner-config", "/unused"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for action contract")
class PlannerLoopTests(unittest.TestCase):
    def setUp(self):
        import test_guard
        self.fixture = test_guard.GuardLoopTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.guard = self.fixture.guard
        self.guard.planner = Mock(max_calls=3, history_limit=8)
        self.guard.planner.plan.return_value = {"vla_instruction": "Close lid before lifting."}
        self.guard.planner_calls = self.guard.planner_failures = 0
        self.guard.planner_latency_s = 0.
        self.guard.executed_history, self.guard.candidate_history = [], []

    def select(self, regenerate):
        f = self.fixture
        return self.guard.select_chunk(0, f.obs, f.chunk(), regenerate, f.bounds, 4)

    def test_planner_instruction_reaches_vla_and_guard_keeps_original_task(self):
        f = self.fixture
        f.scorer([.9, .1, .1, .1])
        def regenerate(obs):
            self.assertEqual(obs["task_descriptions"], "Close lid before lifting.")
            self.assertEqual(obs["episode_seed"], 123)
            return f.chunk(.2)
        self.assertIsNotNone(self.select(regenerate))
        self.assertEqual(f.obs["task_descriptions"], "transport")
        request_context = self.guard.planner.plan.call_args.args[0]
        self.assertEqual(request_context["executed_history"], [])
        self.assertEqual(request_context["guard_feedback"][0]["verdict"], "violated")
        forecasts = f.rows("forecasts.jsonl")
        self.assertEqual(forecasts[0]["input"]["task_instruction"], "transport")
        self.assertEqual(forecasts[0]["input"]["policy_instruction"], "Close lid before lifting.")
        body = OpenJevHTTP("http://unused", "fixture", "proprio_only").request_body(forecasts[0])
        self.assertEqual(body["state"]["task_instruction"], "transport")
        self.assertEqual(body["state"]["policy_instruction"], "Close lid before lifting.")
        self.assertEqual(len(f.rows("planner.jsonl")), 1)
        # Only after an actual env step is an action added to executed history.
        self.assertEqual(self.guard.executed_history, [])
        with patch.object(ShadowEpisode, "after_step"):
            self.guard.after_step(1, None, f.obs)
        self.assertEqual(len(self.guard.executed_history), 1)
        self.assertEqual(self.guard.executed_history[0]["executed_action"],
                         forecasts[0]["input"]["remaining_actions"][0])

    def test_passing_guard_skips_planner(self):
        self.fixture.scorer([.1, .1])
        self.assertIsNotNone(self.select(Mock()))
        self.guard.planner.plan.assert_not_called()

    def test_planner_cannot_bypass_rejection_and_retries_remain_bounded(self):
        self.fixture.scorer([.9, .1] * 4)
        self.assertIsNone(self.select(lambda obs: self.fixture.chunk(.2)))
        self.assertEqual(self.guard.planner.plan.call_count, 3)
        self.assertEqual(self.guard.termination_reason, "guard_retries_exhausted")
        self.assertEqual(self.fixture.rows("forecasts.jsonl"), [])

    def test_planner_failure_or_budget_exhaustion_never_calls_vla(self):
        for budget, error, expected in ((0, None, "planner_budget_exhausted"),
                                         (3, PlannerError("offline"), "planner_error")):
            with self.subTest(expected=expected):
                self.guard.planner.max_calls = budget
                self.guard.planner.plan.side_effect = error
                self.fixture.scorer([.9, .1])
                regenerate = Mock()
                self.assertIsNone(self.select(regenerate))
                regenerate.assert_not_called()
                self.assertEqual(self.guard.termination_reason, expected)
                self.assertEqual(self.fixture.rows("forecasts.jsonl"), [])

    @unittest.skipUnless(os.environ.get("SAFETYJEV_MANIGUARD_ROOT"), "set SAFETYJEV_MANIGUARD_ROOT for runner integration")
    def test_openrouter_repair_through_actual_runner_and_openpi_prompt_mapping(self):
        f, g = self.fixture, self.guard
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "TEST_SECRET", "OPENROUTER_MODEL": "provider/test"}):
            g.planner = OpenRouterPlanner({"include_images": False})
        f.scorer([.9, .1, .1, .1])
        g.last_state, g.last_observation_step = f.obs["states"].tolist(), 0
        root = Path(os.environ["SAFETYJEV_MANIGUARD_ROOT"])
        verify_sources(root)
        tree = ast.parse(instrument((root / "maniguard/eval/benchmark.py").read_text(), "guard_regenerate"))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and ast.unparse(n.test) == "step_idx < cfg.max_steps and (not done)")
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ("_remap_obs_for_openpi", "query_policy")]
        code = compile(ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[])),
                       "openrouter-pinned-loop", "exec")
        prompts, executed = [], []
        def infer(obs):
            prompts.append(obs["prompt"])
            self.assertEqual(obs["episode_seed"], 123)
            return {"actions": f.chunk(.2 if len(prompts) > 1 else .1)}
        def step(action):
            executed.append(action.copy())
            return None, 0., None, None, None
        cfg = SimpleNamespace(execute_horizon=4, max_steps=1, gripper_binarize=True,
                              ik_eef_to_joint=False, save_video=False)
        scope = {
            "cfg": cfg, "EvalConfig": object, "step_idx": 0, "done": False, "success": False,
            "policy": SimpleNamespace(infer=infer), "client_type": "openpi",
            "obs": f.obs.copy(), "os": SimpleNamespace(environ={}), "sj_shadow": g,
            "np": f.np, "action_space": f.bounds, "env": SimpleNamespace(step=step),
            "torch": SimpleNamespace(from_numpy=lambda a: SimpleNamespace(unsqueeze=lambda axis: a)),
            "extract_obs": lambda *args: f.obs.copy(), "robot": None,
            "scene_info": {"prompt": "transport"}, "episode_seed": 123, "total_reward": 0.,
            "ever_contacted": False, "_task_objs": [], "_target_obj": None,
            "monitor": None, "goal_checker": None, "goal_detail": {},
        }
        with patch("safetyjev.planner.urlopen", return_value=response()) as http, patch("sys.stdout", new=io.StringIO()):
            exec(code, scope)
        self.assertEqual(prompts, ["transport", "Close the jar lid before lifting."])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(float(executed[0][0]), .2)
        self.assertEqual(len(g.executed_history), 1)
        self.assertEqual(f.obs["task_descriptions"], "transport")
        self.assertEqual(g.executed_history[0]["policy_instruction"], prompts[1])


if __name__ == "__main__":
    unittest.main()
