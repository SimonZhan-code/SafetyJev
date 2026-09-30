import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class MixtureTests(unittest.TestCase):
    def test_groups_are_stable_and_never_split_by_episode_or_condition(self):
        from safetyjev.data_mixture import assign_group_splits
        groups = [f"jar/task_{i:04d}" for i in range(10)]
        fractions = {"train": .6, "validation": .2, "test": .2}
        a = assign_group_splits(groups, fractions, seed=42)
        b = assign_group_splits(list(reversed(groups)), fractions, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a.values()).count("train"), 6)
        more = assign_group_splits(groups + ["lid/task_0000", "lid/task_0001"], fractions, seed=42)
        self.assertEqual(a, {k: more[k] for k in groups})
        with self.assertRaises(ValueError):
            assign_group_splits(groups, {"train": .8, "test": .8}, seed=42)

    def test_two_family_package_preserves_separate_roots_and_native_targets(self):
        from safetyjev.data_mixture import build_mixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for family in ("jar", "lid"):
                raw = root / family
                raw.mkdir()
                trace = raw / "task_0000/base_ltl.json"
                trace.parent.mkdir()
                trace.write_text(json.dumps({"log": [
                    {"step": 0, "ap": {"upright": True}},
                    {"step": 1, "ap": {"upright": False}},
                    {"step": 2, "ap": {"upright": True}},
                ]}))
                observations = {}
                for view, name in (("overview", "main"), ("wrist", "wrist")):
                    p = trace.with_name(f"base_{name}.mp4")
                    p.write_bytes(family.encode())
                    observations[view] = str(p.relative_to(raw))
                (raw / "manifest.jsonl").write_text(json.dumps({"id": "same-id", "family": family})+"\n")
                gt = root / (family + "-gt.jsonl")
                gt.write_text(json.dumps({
                    "trajectory_id": "same-id", "last_timestep": 2, "ap_trace": str(trace.relative_to(raw)),
                    "ap_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "observations": observations, "video_fps": 30, "control_hz": 20,
                    "metadata": {"family": family, "split_group": family + "/task_0000"},
                })+"\n")
                query = root / (family + "-queries.json")
                query.write_text(json.dumps({"schema_version": 1, "family": family, "queries": [
                    {"id": "tilted", "ap": "upright", "yes_if": False, "question": "Is the object tilted?"}]}))
                sources.append({"name": family, "raw": str(raw), "trajectories": str(gt),
                                "queries": str(query), "license": "test-fixture-only"})
            recipe = {"sources": sources, "seed": 42, "splits": {"train": 1.0},
                      "window": {"history_frames": 2, "frame_stride": 1, "sample_stride": 1}}
            out = root / "package"
            report = build_mixture(recipe, out)
            with (out / "train.jsonl").open() as stream:
                rows = list(map(json.loads, stream))
            self.assertEqual(report["samples"], 4)
            self.assertEqual(len({r["id"] for r in rows}), 4)
            self.assertEqual([r["target"] for r in rows], [[0., 1.], [1., 0.]] * 2)
            self.assertEqual({r["state"]["observation_window"]["resource_id"] for r in rows}, {"jar", "lid"})
            for name in ("jar", "lid"):
                self.assertEqual((out / report["resources"][name]["raw_root"]).resolve(), root / name)
            self.assertFalse((out / "BUILDING").exists())
            self.assertTrue((out / "group_splits.json").exists())
            with self.assertRaises(FileExistsError):
                build_mixture(recipe, out)


if __name__ == "__main__":
    unittest.main()
