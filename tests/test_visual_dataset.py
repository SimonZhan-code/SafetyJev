import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

HAS_AV = importlib.util.find_spec("av") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class VisualDatasetTests(unittest.TestCase):
    @unittest.skipUnless(HAS_AV and shutil.which("ffmpeg"), "PyAV and ffmpeg are needed for the real-video decoder test")
    def test_decoder_seeks_exact_frames_and_preserves_rgb_order(self):
        from safetyjev.visual_dataset import VideoWindowDecoder
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "colors.mp4"
            frames = np.zeros((12, 16, 16, 3), dtype=np.uint8)
            for i in range(12):
                frames[i, :, :, :] = [i * 17, 255 - i * 17, i * 7]
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                            "-video_size", "16x16", "-framerate", "30", "-i", "pipe:0", "-c:v",
                            "libx264rgb", "-crf", "0", "-g", "4", str(video)], input=frames.tobytes(), check=True)
            decoder = VideoWindowDecoder(cache_windows=2)
            actual = decoder.decode(video, [3, 7, 11], expected_fps=30)
            np.testing.assert_array_equal(actual, frames[[3, 7, 11]])
            with self.assertRaises(ValueError):
                decoder.decode(video, [12], expected_fps=30)
            with self.assertRaises(ValueError):
                decoder.decode(video, [2, 1], expected_fps=30)

    def test_collator_preserves_time_camera_and_target_without_metadata_in_inputs(self):
        from safetyjev.visual_dataset import collate_numpy
        image = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        item = {"inputs": {"question": "Is the jar closed?", "observations": {"overview": image, "wrist": image+1}},
                "target": np.array([0., 1.], dtype=np.float32), "sample_id": "a", "metadata": {"condition": "env"}}
        b = collate_numpy([item, {**item, "sample_id": "b"}])
        self.assertEqual(set(b["inputs"]), {"questions", "observations"})
        self.assertEqual(b["inputs"]["observations"]["overview"].shape, (2, 2, 4, 4, 3))
        self.assertEqual(b["targets"].tolist(), [[0., 1.], [0., 1.]])
        self.assertEqual(b["sample_ids"], ["a", "b"])
        with self.assertRaises(ValueError):
            collate_numpy([item, {**item, "inputs": {"question": "x", "observations": {"overview": image[:1], "wrist": image[:1]}}}])

    def test_dataset_resolves_resource_root_and_training_balance_is_explicit(self):
        from safetyjev.visual_dataset import APWindowDataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"; raw.mkdir()
            (raw / "video.mp4").write_bytes(b"fixture")
            (root / "dataset_metadata.json").write_text(json.dumps({"resources": {"one": {"raw_root": "raw"}}}))
            rows = []
            for i, answer in enumerate([0, 0, 1]):
                rows.append({"id": str(i), "kind": "noul", "options": ["no", "yes"], "question": "Is it tilted?",
                             "state": {"observation_window": {"resource_id": "one", "videos": {"overview": "video.mp4", "wrist": "video.mp4"}, "frame_indices": [0, 1], "video_fps": 30}},
                             "target": [1-answer, answer], "metadata": {"family": "jar", "query_id": "tilted"}})
            for split in ("train", "test"):
                (root / (split+".jsonl")).write_text(''.join(json.dumps(r)+"\n" for r in rows))
            dataset = APWindowDataset(root, "train")
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.record(2)["target"], [0, 1])
            weights = dataset.training_weights("query_answer")
            self.assertAlmostEqual(weights[2] / weights[0], 2.)
            requested = dataset.training_weights("query_answer", positive_fraction=.6)
            self.assertAlmostEqual(requested[2] / requested.sum(), .6)
            self.assertEqual(dataset.record(2)["target"], [0, 1])
            for fraction in (-.1, 0, 1, float("nan"), True):
                with self.assertRaises(ValueError):
                    dataset.training_weights("query_answer", positive_fraction=fraction)
            with self.assertRaises(ValueError):
                dataset.training_weights("uniform", positive_fraction=.6)
            self.assertEqual(dataset.media_path(dataset.record(0), "overview"), raw / "video.mp4")
            with self.assertRaises(ValueError):
                APWindowDataset(root, "test").training_weights("query_answer")

    @unittest.skipUnless(HAS_TORCH, "Torch batch test uses the existing behavior environment")
    def test_torch_batch_has_b_t_c_h_w_and_noul_targets(self):
        from safetyjev.visual_dataset import collate_torch
        pixels = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        item = {"inputs": {"question": "Is it closed?", "observations": {"overview": pixels, "wrist": pixels}},
                "target": np.array([1., 0.], dtype=np.float32), "sample_id": "a"}
        batch = collate_torch([item])
        self.assertEqual(tuple(batch["inputs"]["observations"]["overview"].shape), (1, 2, 3, 4, 4))
        self.assertEqual(batch["targets"].tolist(), [[1., 0.]])


if __name__ == "__main__":
    unittest.main()
