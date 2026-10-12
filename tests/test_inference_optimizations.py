import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from safetyjev.agentic_config import IDENTITY, validate_config
from safetyjev.agentic_predictor import NativePredictor
from safetyjev.inference_optimizations import VisionFeatureCache, feature_key, reference_construction

spec = importlib.util.spec_from_file_location('fast_bench', Path(__file__).resolve().parents[1]/'scripts/remote/benchmark-predictor-fast.py')
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)


class OptimizationsTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(Tensor=type('Tensor', (), {}), is_grad_enabled=lambda: False)
        self.patch = patch.dict(sys.modules, torch=self.torch); self.patch.start(); self.addCleanup(self.patch.stop)

    def test_cache_matches_all_arguments_and_clear_discards_features(self):
        backbone = SimpleNamespace(training=False, get_image_features=Mock(side_effect=lambda *a, **k: {'features': object()}))
        cache = VisionFeatureCache(backbone); cache.enabled = True
        first = backbone.get_image_features('pixels', grid=[1, 2, 3], return_dict=True)
        second = backbone.get_image_features('pixels', return_dict=True, grid=[1, 2, 3])
        self.assertIs(first['features'], second['features']); self.assertIsNot(first, second)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        for pixels, grid in [('changed', [1, 2, 3]), ('changed', [1, 3, 2])]:
            backbone.get_image_features(pixels, grid=grid, return_dict=True)
        self.assertEqual(cache.original.call_count, 3)
        cache.clear(); backbone.get_image_features('changed', grid=[1, 3, 2], return_dict=True)
        self.assertEqual(cache.original.call_count, 4)

    def test_failed_forward_does_not_populate_cache(self):
        backbone = SimpleNamespace(training=False, get_image_features=Mock(side_effect=[RuntimeError(), {'ok': True}]))
        cache = VisionFeatureCache(backbone); cache.enabled = True
        with self.assertRaises(RuntimeError): backbone.get_image_features('pixels')
        self.assertIsNone(cache.key)
        backbone.get_image_features('pixels'); self.assertEqual(cache.original.call_count, 2)

    def test_training_grad_or_disabled_cache_never_reuses_features(self):
        backbone = SimpleNamespace(training=False, get_image_features=Mock(return_value={}))
        cache = VisionFeatureCache(backbone)
        for enabled, training, grad in [(False, False, False), (True, True, False), (True, False, True)]:
            cache.enabled = enabled; backbone.training = training; self.torch.is_grad_enabled = lambda: grad
            before = cache.original.call_count
            backbone.get_image_features('pixels'); backbone.get_image_features('pixels')
            self.assertEqual(cache.original.call_count-before, 2)

    def test_unknown_argument_bypasses_cache_and_distinct_structures_hash_differently(self):
        backbone = SimpleNamespace(training=False, get_image_features=Mock(return_value={}))
        cache = VisionFeatureCache(backbone); cache.enabled = True
        for _ in range(2): backbone.get_image_features(object())
        self.assertEqual(cache.original.call_count, 2)
        self.assertNotEqual(feature_key(([1, 2],), {}), feature_key(([2, 1],), {}))
        self.assertNotEqual(feature_key(('12',), {}), feature_key((12,), {}))

    def test_optional_kernels_are_disabled_during_load_and_restored_after_error(self):
        names = ('FusedRMSNormGated', 'causal_conv1d_fn', 'causal_conv1d_update',
                 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule')
        originals = {name: object() for name in names}; qwen = SimpleNamespace(**originals)
        with patch.dict(sys.modules, {'transformers.models.qwen3_5': SimpleNamespace(modeling_qwen3_5=qwen)}):
            with self.assertRaises(RuntimeError):
                with reference_construction():
                    self.assertTrue(all(getattr(qwen, n) is None for n in names)); raise RuntimeError()
        for name, original in originals.items(): self.assertIs(getattr(qwen, name), original)

    def test_config_and_client_reject_silent_backend_substitution(self):
        config = validate_config({'predictor': {'attention_backend': 'fla', 'vision_cache': True}})
        client = NativePredictor(config['predictor'])
        with self.assertRaisesRegex(ValueError, 'optimization mismatch'):
            client.validate_health({**IDENTITY, 'status': 'ready'})
        client.validate_health({**IDENTITY, 'status': 'ready', 'optimizations': {'attention_backend': 'fla', 'vision_cache': True}})
        for value in ({'attention_backend': 'unknown'}, {'vision_cache': 1}):
            with self.assertRaises(ValueError): validate_config({'predictor': value})

    def test_speedup_cannot_pass_when_a_binary_answer_flips(self):
        reference = {'mode': 'torch', 'repeat': 0, 'wall_s': 2, 'peak_allocated_gib': 1,
                     'answers': {'a': {'violated': False, 'logit': -.01}}}
        candidate = copy.deepcopy(reference); candidate.update(mode='fla', wall_s=.5)
        candidate['answers']['a'] = {'violated': True, 'logit': .01}
        summary = bench.summarize([{'measurements': [reference, candidate]}], ['torch', 'fla'])['fla']
        self.assertEqual(summary['speedup'], 4); self.assertFalse(summary['passed'])
        self.assertEqual(summary['acceptance_disagreements'], 1)


if __name__ == '__main__': unittest.main()
