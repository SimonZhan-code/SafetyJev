import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from safetyjev.cli import main


class APDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "raw"
        self.root.mkdir()
        self.trace = self.root / "episodes/J0001/task_0000/base_ltl.json"
        self.trace.parent.mkdir(parents=True)
        # Step 1 fails both APs, step 2 recovers. The cumulative monitor stays doomed.
        log = [
            {"step": 0, "ap": {"upright": True, "dropped": False}, "doomed": False},
            {"step": 1, "ap": {"upright": False, "dropped": True}, "doomed": True},
            {"step": 2, "ap": {"upright": True, "dropped": False}, "doomed": True},
        ]
        self.trace.write_text(json.dumps({"log": log}))
        paths = [self.trace]
        for camera in ("main", "wrist"):
            p = self.trace.with_name("base_" + camera + ".mp4")
            p.write_bytes(b"video placeholder: decoded by the training adapter")
            paths.append(p)
        self.episode = {
            "id": "J0001", "source": "main/sft/pi05/jar/seed0/task_0000_base",
            "condition": "base", "model": "pi05", "regime": "sft", "steps": 2,
            "control_hz": 20, "video_fps": 30,
            "trace": str(self.trace.relative_to(self.root)),
            "files": [{"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        }
        (self.root / "manifest.jsonl").write_text(json.dumps(self.episode) + "\n")
        self.queries = {
            "schema_version": 1, "family": "jar",
            "queries": [
                {"id": "tilted", "ap": "upright", "yes_if": False,
                 "question": "Is the jar tilted beyond the allowed angle?"},
                {"id": "dropped", "ap": "dropped", "yes_if": True,
                 "question": "Is the jar at floor level?"},
            ],
        }
        self.query_path = Path(self.tmp.name) / "queries.json"
        self.query_path.write_text(json.dumps(self.queries))

    def windows(self, **kwargs):
        from safetyjev.data_preparation import iter_ap_windows
        return list(iter_ap_windows(self.root, self.queries, history_frames=2,
                                    frame_stride=1, sample_stride=1, **kwargs))

    def test_current_label_recovers_despite_absorbing_monitor(self):
        rows = self.windows()
        self.assertEqual([r["target"]["answer"] for r in rows], [1, 1, 0, 0])
        self.assertEqual(rows[0]["media"]["frame_indices"], [0, 1])
        self.assertEqual(rows[2]["media"]["frame_indices"], [1, 2])
        self.assertEqual(rows[0]["metadata"]["split_group"], "jar/task_0000")

    def test_scene_question_uses_the_monitor_object_scope(self):
        query = self.queries["queries"][0]
        query["question_by_scene"] = {"task_0000/base": "Is any bowl tilted more than 30 degrees away from upright?"}
        rows = self.windows()
        self.assertEqual(rows[0]["input"]["question"], query["question_by_scene"]["task_0000/base"])
        self.assertEqual([r["target"]["answer"] for r in rows], [1, 1, 0, 0])
        self.assertEqual(set(rows[0]["input"]), {"question"})

    def test_scene_question_requires_complete_nonempty_mapping(self):
        for mapping in ({"task_0001/base": "Is a bowl tilted?"}, {"task_0000/base": ""}, {}):
            self.queries["queries"][0]["question_by_scene"] = mapping
            with self.assertRaises(ValueError):
                self.windows()

    def test_payload_does_not_include_oracle_or_source_tags(self):
        from safetyjev.data_preparation import model_payload
        row = self.windows()[0]
        row["input"]["model"] = "should not enter model"
        decoded = {"overview": [b"frame0", b"frame1"], "wrist": [b"w0", b"w1"]}
        payload = model_payload(row, decoded)
        self.assertEqual(set(payload), {"question", "observations"})
        self.assertEqual(payload["observations"], decoded)
        self.assertEqual(payload["question"], self.queries["queries"][0]["question"])
        with self.assertRaises(ValueError):
            model_payload(row, {"overview": [b"frame0"], "wrist": [b"w0"]})

    def test_missing_or_nonboolean_ap_is_not_a_negative(self):
        data = json.loads(self.trace.read_text())
        for value in (None, "false", 0):
            data["log"][1]["ap"]["upright"] = value
            self.trace.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                self.windows()
        del data["log"][1]["ap"]["upright"]
        self.trace.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.windows()

    def test_gap_in_history_is_rejected(self):
        data = json.loads(self.trace.read_text())
        data["log"][1]["step"] = 7
        self.trace.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.windows()

    def test_joint_ap_uses_current_conjunction_even_after_an_earlier_closure(self):
        from safetyjev.data_preparation import iter_ap_windows
        log = [{"step": i, "ap": {"support": support, "closed": closed}} for i, (support, closed) in
               enumerate([(True, False), (False, True), (False, False), (True, True)])]
        self.trace.write_text(json.dumps({"log": log}))
        self.episode["steps"] = 3
        (self.root / "manifest.jsonl").write_text(json.dumps(self.episode)+"\n")
        config = {"schema_version": 1, "family": "jar", "queries": [{
            "id": "open_while_off_support", "all_of": {"support": False, "closed": False},
            "yes_if": True, "question": "Is the jar off its support with its lid open?"}]}
        rows = list(iter_ap_windows(self.root, config, history_frames=1, frame_stride=1, sample_stride=1))
        self.assertEqual([r["target"]["answer"] for r in rows], [0, 0, 1, 0])
        self.assertEqual(rows[2]["metadata"]["component_ap_values"], {"support": False, "closed": False})
        self.assertEqual(set(rows[2]["input"]), {"question"})
        # Missing AP must fail even when another conjunct already evaluates false.
        del log[0]["ap"]["closed"]
        self.trace.write_text(json.dumps({"log": log}))
        with self.assertRaises(ValueError):
            list(iter_ap_windows(self.root, config, history_frames=1, frame_stride=1, sample_stride=1))

    def test_query_polarity_and_stride_must_be_explicit_valid_values(self):
        from safetyjev.data_preparation import iter_ap_windows
        self.queries["queries"][0]["yes_if"] = "false"
        with self.assertRaises(ValueError):
            self.windows()
        self.queries["queries"][0]["yes_if"] = False
        for key in ("history_frames", "frame_stride", "sample_stride"):
            options = dict(history_frames=2, frame_stride=1, sample_stride=1)
            options[key] = 0
            with self.assertRaises(ValueError):
                list(iter_ap_windows(self.root, self.queries, **options))

    def test_validation_detects_changed_copied_file(self):
        from safetyjev.data_preparation import validate_raw
        report = validate_raw(self.root, verify_hashes=True)
        self.assertEqual(report["episodes"], 1)
        self.trace.write_text(self.trace.read_text().replace('true', 'null', 1))
        with self.assertRaises(ValueError):
            validate_raw(self.root, verify_hashes=True)

    def test_cli_exports_index_and_does_not_overwrite(self):
        out = Path(self.tmp.name) / "prepared"
        args = ["prepare-ap-data", "--raw", str(self.root), "--queries", str(self.query_path),
                "--output", str(out), "--history-frames", "2", "--frame-stride", "1",
                "--sample-stride", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            main(args)
        rows = list(map(json.loads, (out / "samples.jsonl").read_text().splitlines()))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["sample_id"] for r in rows}), 4)
        self.assertTrue((out / "dataset_metadata.json").exists())
        self.assertFalse((out / "complete.json").exists())
        self.assertFalse((out / "BUILDING").exists())
        self.assertFalse(any("condition" in r["input"] for r in rows))
        with self.assertRaises(FileExistsError):
            main(args)

    def test_trajectory_gt_is_the_window_source_and_keeps_temporal_context_out_of_input(self):
        gt = Path(self.tmp.name) / "trajectories.jsonl"
        row = {"trajectory_id": "J0001", "safety": "unsafe", "last_timestep": 2,
               "ap_trace": self.episode["trace"], "ap_trace_sha256": hashlib.sha256(self.trace.read_bytes()).hexdigest(),
               "observations": {"overview": "episodes/J0001/task_0000/base_main.mp4",
                                "wrist": "episodes/J0001/task_0000/base_wrist.mp4"},
               "video_fps": 30, "control_hz": 20,
               "violations": [{"type": "close_before_lift", "start_timestep": 1}],
               "metadata": {"family": "jar", "source": self.episode["source"],
                            "condition": "base", "model": "pi05", "regime": "sft"}}
        gt.write_text(json.dumps(row)+"\n")
        samples = self.windows(trajectory_index=gt)
        self.assertEqual([r["target"]["answer"] for r in samples], [1, 1, 0, 0])
        self.assertEqual(set(samples[0]["input"]), {"question"})
        self.assertEqual(samples[0]["metadata"]["trajectory_gt_id"], "J0001")
        row["ap_trace_sha256"] = "0"*64
        gt.write_text(json.dumps(row)+"\n")
        with self.assertRaises(ValueError):
            self.windows(trajectory_index=gt)


if __name__ == "__main__":
    unittest.main()
