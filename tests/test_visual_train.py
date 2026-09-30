import unittest
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile


@unittest.skipUnless(importlib.util.find_spec("torch"), "visual training tests require the training environment")
class VisualTrainTests(unittest.TestCase):
    def test_train_validation_test_pipeline_has_no_calibration_dependency(self):
        from unittest.mock import patch
        import numpy as np
        import torch
        from safetyjev.visual_train import run

        requested = []
        class Dataset:
            def __init__(self, package, split):
                requested.append(split)
                if split not in ("train", "validation", "test"):
                    raise AssertionError("Unexpected split: " + split)
            def __len__(self): return 2
            def __getitem__(self, i):
                image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
                return {"inputs": {"question": "Is it tilted?", "observations": {"overview": image, "wrist": image}},
                        "target": np.array([1-i, i], dtype=np.float32), "sample_id": str(i), "query_id": "tilted"}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.head=torch.nn.Linear(1,1)
                self.backbone=torch.nn.Module();self.backbone.visual=torch.nn.Linear(1,1)
                self.backbone.visual.requires_grad_(False);self.temperature=1.
            def forward(self, questions, observations): return torch.zeros((len(questions),2))
            def set_temperature(self, value): self.temperature=value
            def save(self, path):
                path.mkdir();(path/'model.json').write_text(json.dumps({'temperature': self.temperature}))
        def train(model, loader, config, output, **kwargs):
            kwargs['validation_fn'](model,1)
            return {'status':'completed','best_checkpoint':str(output/'checkpoints/step-00000001')}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';package.mkdir()
            (package/'dataset_metadata.json').write_text('{}')
            config={'model':{'dtype':'float32'},'data':{'package':str(package),'num_workers':0,'batch_size':2,'balance':'uniform'},
                    'seed':42,'training':{},'evaluation':{'max_batches':None}}
            with patch('safetyjev.visual_train.verify_package',return_value={'window':{'history_frames':1},'file_sha256':{}}), \
                 patch('safetyjev.visual_train.APWindowDataset',Dataset), \
                 patch('jev.visual_model.VisualDecisionModel.from_pretrained',side_effect=lambda **kwargs:Model()), \
                 patch('jev.visual_model.VisualDecisionModel.load',side_effect=lambda *args,**kwargs:Model()), \
                 patch('jev.visual_training.fit_updates',side_effect=train):
                report=run(config,root/'run',device='cpu')
            self.assertEqual(requested,['train','validation','test'])
            self.assertEqual(report['temperature'],1.)
            self.assertNotIn('calibration',report)
            self.assertFalse((root/'run/final/calibration.jsonl').exists())
            self.assertTrue((root/'run/final/test.jsonl').is_file())

    def test_epoch_sampler_resumes_at_batch_boundary(self):
        from safetyjev.visual_train import EpochBatchSampler
        full = list(EpochBatchSampler(7, 2, seed=17, epoch=0))
        tail = list(EpochBatchSampler(7, 2, seed=17, epoch=0, start_batch=2))
        self.assertEqual(tail, full[2:])
        self.assertEqual(sorted(i for b in full for i in b), list(range(7)))
        self.assertNotEqual(full, list(EpochBatchSampler(7, 2, seed=17, epoch=1)))

    def test_balanced_sampler_is_reproducible_when_resumed(self):
        from safetyjev.visual_train import EpochBatchSampler
        full = list(EpochBatchSampler(5, 2, seed=7, epoch=3, weights=[1,1,1,1,20]))
        tail = list(EpochBatchSampler(5, 2, seed=7, epoch=3, start_batch=1, weights=[1,1,1,1,20]))
        self.assertEqual(tail, full[1:])

    def test_confusion_counts_use_yes_as_positive_and_missing_class_is_explicit(self):
        from safetyjev.visual_train import confusion
        self.assertEqual(confusion([[1,0],[1,0],[0,1],[0,1]], [.1,.8,.2,.9]),
                         {"tn":1,"fp":1,"fn":1,"tp":1,"yes_precision":.5,"yes_recall":.5})
        self.assertIsNone(confusion([[1,0]], [.1])["yes_recall"])

    def test_changed_pixels_are_rejected_before_training_or_resume(self):
        from safetyjev.visual_train import verify_package
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            video=raw/'camera.mp4';video.write_bytes(b'abcd')
            manifest=raw/'manifest.jsonl';manifest.write_text(json.dumps({"files":[{
                "path":"camera.mp4","bytes":4,"sha256":hashlib.sha256(b'abcd').hexdigest()}]})+'\n')
            split=root/'train.jsonl';split.write_text('{}\n')
            (root/'dataset_metadata.json').write_text(json.dumps({"file_sha256":{"train":hashlib.sha256(split.read_bytes()).hexdigest()},
                "resources":{"raw":{"raw_root":"raw","raw_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}}}))
            verify_package(root)
            video.write_bytes(b'efgh')
            with self.assertRaisesRegex(ValueError,'Raw'):
                verify_package(root)

    def test_failed_finalization_can_be_retried_without_partial_publication(self):
        from safetyjev.visual_train import publish_final
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def fail(stage):
                (stage/'partial').write_text('partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):publish_final(root,fail)
            self.assertFalse((root/'final').exists())
            def succeed(stage):
                (stage/'report.json').write_text('{}')
                return 'done'
            self.assertEqual(publish_final(root,succeed),'done')
            self.assertTrue((root/'final/report.json').exists())


if __name__ == "__main__":
    unittest.main()
