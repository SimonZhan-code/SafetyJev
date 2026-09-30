import unittest


class OpenJevDataTests(unittest.TestCase):
    def sample(self, answer):
        return {"sample_id": "jar:example", "input": {"question": "Is the jar tilted?"},
                "media": {"videos": {"overview": "overview.mp4", "wrist": "wrist.mp4"},
                          "frame_indices": [0, 1], "video_fps": 30, "control_hz": 20},
                "target": {"answer": answer},
                "metadata": {"split_group": "jar/task_0000", "model": "private-source-tag",
                             "condition": "env", "ap_value": False}}

    def test_noul_schema_and_no_yes_target_order(self):
        from safetyjev.openjev_data import to_openjev_record
        provenance = {"input_sha256": "a"*64, "source_url": "file:///fixture",
                      "license": "test-fixture-only", "split_policy": "test-groups"}
        for answer, expected in [(0, [1.0, 0.0]), (1, [0.0, 1.0])]:
            r = to_openjev_record(self.sample(answer), split="train", source="fixture", provenance=provenance)
            self.assertEqual(set(r), {"id", "group_id", "split", "source", "state", "question",
                                      "kind", "options", "target", "metadata"})
            self.assertEqual(r["options"], ["no", "yes"])
            self.assertEqual(r["kind"], "noul")
            self.assertEqual(r["target"], expected)
            self.assertNotIn("model", r["state"])
            self.assertNotIn("ap_value", r["state"])
            self.assertEqual(r["metadata"]["provenance"]["original_id"], "jar:example")

    def test_visual_adapter_passes_pixels_not_resource_ids(self):
        from safetyjev.openjev_data import to_openjev_record, visual_model_payload
        r = to_openjev_record(self.sample(1), split="validation", source="fixture", provenance={
            "input_sha256": "b"*64, "source_url": "file:///fixture", "license": "test", "split_policy": "test"})
        images = {"overview": [b"image0", b"image1"], "wrist": [b"image0", b"image1"]}
        payload = visual_model_payload(r, images)
        self.assertEqual(payload, {"question": "Is the jar tilted?", "observations": images})
        with self.assertRaises(ValueError):
            visual_model_payload(r, {"overview": [b"image0"], "wrist": [b"image0"]})

    def test_unassigned_split_or_missing_provenance_cannot_look_training_ready(self):
        from safetyjev.openjev_data import to_openjev_record
        for split in (None, "unassigned", "ID"):
            with self.assertRaises(ValueError):
                to_openjev_record(self.sample(1), split=split, source="fixture", provenance={})
        with self.assertRaises(ValueError):
            to_openjev_record(self.sample(1), split="train", source="fixture", provenance={})


if __name__ == "__main__":
    unittest.main()
