import importlib
import unittest

class JudgeEntrypointTests(unittest.TestCase):
    def test_judge_entrypoints_exist(self):
        for name in ('commands', 'capture', 'dataset', 'train', 'eval'):
            self.assertIsNotNone(importlib.util.find_spec('safetyjev.predictor_judge_' + name))

    def test_resume_identity_covers_base_visual_implementation_and_metrics(self):
        from safetyjev import predictor_judge_train as t
        self.assertTrue(hasattr(t,'training_source_hashes'))
        sources=t.training_source_hashes()
        self.assertIn('visual_model.py',sources)
        self.assertIn('metrics.py',sources)
        self.assertIn('predictor_judge_sampling.py',sources)
