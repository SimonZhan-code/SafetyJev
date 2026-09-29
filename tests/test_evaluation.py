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


class ReportTests(unittest.TestCase):
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
