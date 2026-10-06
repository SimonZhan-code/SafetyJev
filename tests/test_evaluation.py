import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safetyjev.labels import label_forecasts
from safetyjev.metrics import binary_metrics, evaluate
from safetyjev.predictors import OpenJevHTTP
from safetyjev.cli import main
from safetyjev.io import append_jsonl, write_json


def forecast(start=0, horizon=3, cid="a", fid="f"):
    return {"forecast_id": fid, "constraint_id": cid, "start_step": start,
            "end_step": start + horizon,
            "input": {"remaining_actions": [[0] * 8 for _ in range(horizon)],
                      "robot_state": [0] * 8, "past_robot_states": [],
                      "task_instruction": "transport", "constraint": {"ltl": "G !bad"},
                      "action_frequency_hz": 20, "action_convention": "absolute",
                      "images": {"overview": "SECRET_IMAGE_PATH"}}}


def oracle(values):
    return [{"step": i, "valid": value is not None,
             "violated": {"a": value, "__all__": value}} for i, value in enumerate(values)]


class LabelTests(unittest.TestCase):
    def label(self, values, **kw):
        return label_forecasts([forecast(**kw)], oracle(values))[0]

    def test_exact_end_violation(self):
        row = self.label([False, False, False, True])
        self.assertEqual((row["label"], row["first_violation_step"]), (1, 3))

    def test_later_violation_does_not_label_earlier_chunk(self):
        self.assertEqual(self.label([False, False, False, False, True])["label"], 0)

    def test_already_violated_is_excluded(self):
        self.assertEqual(self.label([True, True, True, True])["label_reason"], "already_violated")

    def test_unknown_initial_state_is_not_negative(self):
        self.assertEqual(self.label([None, False, False, False])["label_reason"], "invalid_start_monitor")

    def test_early_success_or_crash_is_censored(self):
        self.assertIsNone(self.label([False, False])["label"])

    def test_observed_positive_survives_later_truncation(self):
        self.assertEqual(self.label([False, True])["label"], 1)

    def test_missing_monitor_step_cannot_be_bridged(self):
        self.assertIsNone(self.label([False, None, True, True])["label"])

    def test_nonzero_start_uses_remaining_horizon(self):
        self.assertEqual(self.label([False, False, False, True], start=1, horizon=2)["label"], 1)

    def test_missing_constraint_is_unknown(self):
        self.assertIsNone(self.label([False] * 4, cid="missing")["label"])

    def test_bad_shape_and_duplicate_ids_rejected(self):
        row = forecast()
        row["end_step"] = 5
        with self.assertRaises(ValueError):
            label_forecasts([row], oracle([False] * 6))
        with self.assertRaises(ValueError):
            label_forecasts([forecast(), forecast()], oracle([False] * 4))
        with self.assertRaises(ValueError):
            label_forecasts([forecast()], oracle([False] * 4) + oracle([False]))


class MetricTests(unittest.TestCase):
    def test_confusion_and_ranking(self):
        result = binary_metrics([(1, .9), (0, .7), (1, .6), (0, .1)], .65)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(result["auroc"], .75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_ties_are_order_invariant(self):
        result = binary_metrics([(1, .5), (0, .5)], .5)
        self.assertEqual(result["auroc"], .5)
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result, binary_metrics([(0, .5), (1, .5)], .5))

    def test_single_class_and_empty_denominators(self):
        result = binary_metrics([(0, .1)], .5)
        self.assertIsNone(result["violation_recall"])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(binary_metrics([], .5)["accuracy"])

    def test_missing_failed_predictions_stay_in_coverage(self):
        labels = label_forecasts([forecast(fid=str(i)) for i in range(3)], oracle([False] * 4))
        result = evaluate(labels, [{"forecast_id": "0", "score": .1},
                                   {"forecast_id": "1", "score": None, "error": "timeout"}])
        self.assertEqual(result["prediction_coverage"], 1 / 3)
        self.assertEqual(result["failed_predictions"], 1)
        self.assertEqual(result["missing_predictions"], 1)

    def test_invalid_scores_duplicate_unknown_ids(self):
        labels = label_forecasts([forecast()], oracle([False] * 4))
        for score in (math.nan, math.inf, -.1, 1.1, True):
            with self.assertRaises(ValueError):
                evaluate(labels, [{"forecast_id": "f", "score": score}])
        for rows in ([{"forecast_id": "unknown", "score": .5}],
                     [{"forecast_id": "f", "score": .5}] * 2):
            with self.assertRaises(ValueError):
                evaluate(labels, rows)

    def test_threshold_one_and_reliability_bins(self):
        result = binary_metrics([(1, 1.), (0, 0.)], 1.)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["reliability_bins"][-1]["count"], 1)


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.predictor = OpenJevHTTP("http://localhost/v1/systemone", "test", "proprio_only")

    def test_oracle_and_images_never_enter_text_request(self):
        row = forecast()
        row["oracle"] = "SECRET_FUTURE"
        row["input"]["monitor_state"] = "SECRET_MONITOR"
        body = json.dumps(self.predictor.request_body(row))
        self.assertNotIn("SECRET", body)
        self.assertEqual(self.predictor.request_body(row)["questions"]["violation"]["type"], "noul")

    def test_response_and_failure(self):
        with patch("safetyjev.predictors.urlopen", return_value=io.BytesIO(b'{"answers":{"violation":{"noul":0.8}}}')):
            self.assertEqual(self.predictor.score(forecast())["score"], .8)
        for body in (b'{}', b'{"answers":{"violation":{"noul":true}}}',
                     b'{"answers":{"violation":{"noul":2}}}'):
            with patch("safetyjev.predictors.urlopen", return_value=io.BytesIO(body)):
                self.assertIsNone(self.predictor.score(forecast())["score"])
        with patch("safetyjev.predictors.urlopen", side_effect=TimeoutError("fixture")):
            self.assertIn("TimeoutError", self.predictor.score(forecast())["error"])

    def test_no_silent_multimodal_claim(self):
        with self.assertRaises(ValueError):
            OpenJevHTTP("http://localhost", "test", "multimodal")


class ReplayTests(unittest.TestCase):
    def args(self, episode):
        return ["predict", "--episodes", str(episode), "--name", "filtered",
                "--endpoint", "http://localhost/v1/systemone", "--model-id", "fixture",
                "--predictor-revision", "pinned", "--input-mode", "proprio_only", "--eligible-only"]

    def test_eligible_replay_preserves_metrics_and_blinds_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp)
            write_json(episode / "episode.json", {"episode_id": "e"})
            write_json(episode / "complete.json", {})
            rows = [forecast(0, 1, fid="negative"), forecast(1, 1, fid="positive"),
                    forecast(2, 1, fid="excluded")]
            truth = oracle([False, False, True, True])
            for row in rows:
                append_jsonl(episode / "forecasts.jsonl", row)
            for row in truth:
                append_jsonl(episode / "oracle.jsonl", row)
            def score(row):
                self.assertEqual(row, next(r for r in rows if r["forecast_id"] == row["forecast_id"]))
                self.assertNotIn("label", row)
                return {"forecast_id": row["forecast_id"], "score": .8}
            with patch("safetyjev.cli.OpenJevHTTP.score", side_effect=score) as mocked:
                main(self.args(episode))
            self.assertEqual([call.args[0]["forecast_id"] for call in mocked.call_args_list],
                             ["negative", "positive"])
            filtered = [json.loads(x) for x in (episode / "filtered.jsonl").read_text().splitlines()]
            labels = label_forecasts(rows, truth)
            result = evaluate(labels, filtered)
            self.assertEqual(result["missing_predictions"], 0)
            self.assertEqual(result["prediction_coverage"], 1)
            self.assertEqual(result["micro"], evaluate(labels, filtered + [{"forecast_id": "excluded", "score": .8}])["micro"])
            self.assertEqual(json.loads((episode / "filtered.meta.json").read_text())["selection"], "eligible_windows")

    def test_incomplete_replay_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp)
            write_json(episode / "episode.json", {"episode_id": "e"})
            with self.assertRaisesRegex(ValueError, "complete episodes"):
                main(self.args(episode))
            self.assertFalse((episode / "filtered.meta.json").exists())


class ReportTests(unittest.TestCase):
    def test_mixed_execution_modes_or_guard_settings_are_not_pooled(self):
        for second in ({"mode": "shadow_no_intervention"},
                       {"mode": "guard_regenerate", "guard": {"threshold": .7}},
                       {"mode": "guard_regenerate", "guard": {"threshold": .5},
                        "planner": {"model": "different/planner"}}):
            with self.subTest(second=second), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for name, execution in (("a", {"mode": "guard_regenerate", "guard": {"threshold": .5}}),
                                        ("b", second)):
                    path = root / name
                    path.mkdir()
                    write_json(path / "episode.json", {"episode_id": name, **execution})
                    write_json(path / "complete.json", {})
                    append_jsonl(path / "oracle.jsonl", oracle([False])[0])
                with self.assertRaisesRegex(ValueError, "execution modes or guard configurations"):
                    main(["report", "--episodes", str(root), "--output", str(root / "report.json")])

    def test_guard_stop_with_no_executed_candidates_is_in_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "episode.json", {"episode_id": "e", "mode": "guard_regenerate"})
            write_json(root / "complete.json", {})
            outcome = {"steps": 0, "status": "completed", "success": False,
                       "safetyjev_guard": {"termination_reason": "guard_retries_exhausted"}}
            write_json(root / "maniguard_result.json", outcome)
            append_jsonl(root / "oracle.jsonl", oracle([False])[0])
            with patch("sys.stdout", new=io.StringIO()):
                main(["report", "--episodes", str(root), "--output", str(root / "report.json")])
            result = json.loads((root / "report.json").read_text())
            self.assertEqual(len(result["episodes"]), 1)
            self.assertEqual(result["episodes"][0]["outcome"], outcome)
            self.assertEqual(result["eligible_labels"], 0)

    def test_global_query_cannot_change_constraint_classification_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "episode.json", {"episode_id": "e", "pipeline": "jar_transport",
                                                "scene_name": "task_0000/base"})
            write_json(root / "complete.json", {})
            write_json(root / "prediction.meta.json", {"model_id": "fixture", "mode": "synthetic"})
            for row in oracle([False, False, False, True]):
                append_jsonl(root / "oracle.jsonl", row)
            for cid, fid, score in (("a", "constraint", .1), ("__all__", "global", .9)):
                append_jsonl(root / "forecasts.jsonl", forecast(cid=cid, fid=fid))
                append_jsonl(root / "predictions.jsonl", {"forecast_id": fid, "score": score})
            with patch("sys.stdout", new=io.StringIO()):
                main(["report", "--episodes", str(root), "--output", str(root / "report.json")])
            report = json.loads((root / "report.json").read_text())
            for metrics in (report, report["episodes"][0]["metrics"], report["by_family_level"]["jar_transport/base"]):
                self.assertEqual(metrics["micro"]["n"], 1)
                self.assertEqual(metrics["micro"]["fn"], 1)
                self.assertEqual(metrics["micro"]["tp"], 0)
                self.assertEqual(metrics["global_task_forecast"]["micro"]["tp"], 1)
                self.assertNotIn("__all__", metrics["by_constraint"])

    def test_report_from_recorded_fixture_and_missing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = root / "episode"
            episode.mkdir()
            write_json(episode / "episode.json", {"episode_id": "e", "pipeline": "jar_transport", "scene_name": "task_0000/base"})
            write_json(episode / "complete.json", {})
            for row in oracle([False, False, True]):
                append_jsonl(episode / "oracle.jsonl", row)
            append_jsonl(episode / "forecasts.jsonl", forecast(cid="__all__"))
            append_jsonl(episode / "predictions.jsonl", {"forecast_id": "f", "score": .8})
            write_json(episode / "prediction.meta.json", {"model_id": "fixture", "mode": "synthetic"})
            missing = root / "incomplete"
            missing.mkdir()
            write_json(missing / "episode.json", {"episode_id": "missing"})
            with patch("sys.stdout", new=io.StringIO()):
                main(["report", "--episodes", str(root), "--output", str(root / "report.json")])
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["global_task_forecast"]["micro"]["tp"], 1)
            self.assertEqual(report["incomplete_episode_ids"], ["missing"])
            self.assertEqual(report["prediction_metadata"][0]["mode"], "synthetic")


if __name__ == "__main__":
    unittest.main()
