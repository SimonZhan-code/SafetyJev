"""Per-action classification: cadence, alignment, failures, and upstream wiring."""
import ast
from contextlib import contextmanager, ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from safetyjev.io import read_jsonl, write_json
from safetyjev.maniguard import instrument, verify_sources
from safetyjev.visual_runtime import VisualClassificationEpisode, per_step_config, report


class ConfigTests(unittest.TestCase):
    def test_new_capture_defaults_to_every_step_and_rejects_sparse_configs(self):
        config = {'queries': []}
        self.assertEqual(per_step_config(config)['sample_stride'], 1)
        self.assertNotIn('sample_stride', config)
        self.assertEqual(per_step_config({'sample_stride': 1})['sample_stride'], 1)
        for stride in (8, 2, 0, -1, True, 1.0, '1', None):
            with self.subTest(stride=stride), self.assertRaisesRegex(ValueError, 'sample_stride=1'):
                per_step_config({'sample_stride': stride})

    def test_saved_dense_and_historical_sparse_reports_check_expected_states(self):
        for stride, steps in ((1, list(range(11))), (8, [0, 8])):
            with self.subTest(stride=stride), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ep = root / 'episode'
                ep.mkdir()
                write_json(ep/'episode.json', {'scene_name': 'fixture', 'visual_classifier': {
                    'sample_stride': stride, 'queries': [{'id': 'bad'}]}})
                write_json(ep/'complete.json', {'status': 'completed', 'monitor_valid': True, 'final_step': 10})
                write_json(ep/'maniguard_result.json', {'steps': 10})
                labels = [{'sample_id': str(s), 'step': s, 'labels': {'bad': 0}} for s in steps]
                (ep/'classification-labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
                (ep/'classification-predictions.jsonl').write_text('')
                result = report(root, root/'report.json')
                self.assertEqual(result['expected_classifications'], len(steps))
                self.assertEqual(result['failed_or_missing_classifications'], len(steps))
                self.assertEqual(result['coverage'], 0)
                (ep/'classification-labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels[:-1]))
                with self.assertRaisesRegex(ValueError, 'label steps'):
                    report(root, root/'report.json')


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required for simulator observation contract')
class PerStepTests(unittest.TestCase):
    @contextmanager
    def episode(self, root, fail_step=None):
        import numpy as np
        scene_path = root/'scene.json'
        scene_path.write_text('{}')
        scene = {'name': 'fixture', 'scene_file': str(scene_path), 'ltl_safety': {
            'constraints': [{'id': 'invariant', 'ltl': 'G !bad'}],
            'propositions': {'bad': {}}, 'combined_ltl': 'G !bad'}}
        cfg = SimpleNamespace(state_mode='joint', ik_eef_to_joint=False, action_dim=8,
                              action_frequency=20, gripper_binarize=True)
        def observation(step):
            return {'states': np.full(16, step, dtype=float), 'task_descriptions': 'transport',
                    'overview_image': np.full((1, 1, 3), step, dtype=np.uint8),
                    'wrist_images': np.full((1, 1, 3), step, dtype=np.uint8)}
        monitor = SimpleNamespace(_monitor=True, violated=False, _ltl_log=[])
        def advance(step):
            bad = step == 3  # A one-step violation entirely between chunk boundaries.
            monitor.violated |= bad
            monitor._ltl_log.append({'step': step, 'ap': {'bad': bad}})
        monitor.step = advance
        monitor.step(0)
        machine = SimpleNamespace(ap_list=['bad'], reset=lambda: None,
                                  step=lambda ap: {'doomed': ap['bad']})
        writer = SimpleNamespace(imwrite=lambda path, array: Path(path).write_bytes(array.tobytes()))
        query = {'id': 'bad', 'ap': 'bad', 'yes_if': True, 'question': 'Is the object tilted?'}
        options = {'output': str(root/'episodes'), 'provenance': {}, 'recheck_every': 0,
                   'visual_config': {'sample_stride': 1, 'queries': [query],
                                     'endpoint': 'http://fixture', 'checkpoint_revision': 'fixture'}}
        requests = []
        def classify(request, **kwargs):
            import base64
            body = json.loads(request.data)
            self.assertEqual(set(body), {'images', 'questions'})
            self.assertEqual(body['questions'], [{'id': 'bad', 'question': query['question']}])
            overview = base64.b64decode(body['images']['overview'])
            self.assertEqual(overview, base64.b64decode(body['images']['wrist']))
            step = overview[0]
            requests.append(step)
            if step == fail_step:
                raise TimeoutError('fixture request timeout')
            score = .9 if step == 3 else .1
            return io.StringIO(json.dumps({'checkpoint_revision': 'fixture',
                'answers': {'bad': {'raw_yes': score, 'calibrated_yes': score,
                                    'logit': 2. if step == 3 else -2.}},
                'server_latency_s': .01, 'input_tokens': 1}))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', {
                'imageio': SimpleNamespace(v2=writer), 'imageio.v2': writer,
                'maniguard.utils.ltl_utils': SimpleNamespace(LTLMonitor=lambda formula: machine)}))
            stack.enter_context(patch('safetyjev.capture.OPTIONS', options))
            stack.enter_context(patch('safetyjev.visual_runtime.urlopen', side_effect=classify))
            observer = VisualClassificationEpisode(scene, cfg, monitor, observation(0), 123)
            yield observer, cfg, monitor, observation, requests

    def test_each_resulting_state_is_scored_and_timeout_does_not_skip_later_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.episode(root, fail_step=4) as (observer, cfg, monitor, observation, requests):
                for step in range(1, 11):
                    monitor.step(step)
                    observer.after_step(step, monitor, observation(step))
                observer.finish({'steps': 10, 'status': 'completed', 'success': False})
                self.assertEqual(requests, list(range(11)))
                labels = read_jsonl(observer.directory/'classification-labels.jsonl')
                self.assertEqual([r['step'] for r in labels], list(range(11)))
                self.assertEqual([r['labels']['bad'] for r in labels], [0, 0, 0, 1] + [0]*7)
                self.assertTrue(monitor.violated)  # Current truth recovers despite cumulative rejection.
                result = report(root/'episodes', root/'report.json')
                self.assertEqual(result['expected_classifications'], 11)
                self.assertEqual(result['failed_or_missing_classifications'], 1)
                self.assertEqual(result['raw']['micro']['tp'], 1)
                self.assertEqual(result['raw']['micro']['tn'], 9)
                self.assertEqual(observer.classification_failures, 1)
                meta = json.loads((observer.directory/'episode.json').read_text())
                self.assertEqual(meta['classification_sampling']['sample_stride'], 1)

    def test_invalid_monitor_state_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.episode(Path(tmp)) as (observer, cfg, monitor, observation, requests):
                # No monitor advance for action 1: there is no aligned oracle label.
                with self.assertRaisesRegex(ValueError, 'invalid aligned monitor'):
                    observer.after_step(1, monitor, observation(1))
                self.assertTrue(observer.invalid)
                self.assertEqual(requests, [0])

    @unittest.skipUnless(os.environ.get('SAFETYJEV_MANIGUARD_ROOT'),
                         'set SAFETYJEV_MANIGUARD_ROOT for pinned-runner test')
    def test_actual_runner_scores_inside_chunks_and_before_early_success(self):
        import numpy as np
        upstream = Path(os.environ['SAFETYJEV_MANIGUARD_ROOT'])
        verify_sources(upstream)
        source = instrument((upstream/'maniguard/eval/benchmark.py').read_text(), 'visual_classification')
        loop = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.While)
                    and ast.unparse(n.test) == 'step_idx < cfg.max_steps and (not done)')
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                       'pinned-classification-loop', 'exec')
        for max_steps, stop_at in ((10, None), (16, 5)):
            with self.subTest(max_steps=max_steps, stop_at=stop_at), tempfile.TemporaryDirectory() as tmp:
                with self.episode(Path(tmp)) as (observer, cfg, monitor, observation, requests):
                    cfg.execute_horizon = 8
                    cfg.max_steps = max_steps
                    cfg.save_video = False
                    cfg.success_hold_steps = 1
                    executed = []
                    def step(action):
                        executed.append(action.copy())
                        return None, 0., None, None, None
                    chunk = np.full((16, 8), .2, dtype=np.float32)
                    original = chunk.copy()
                    policy = Mock(return_value=chunk)
                    goal = (SimpleNamespace(check=lambda env: (len(executed) >= stop_at, {}))
                            if stop_at else None)
                    scope = {'cfg': cfg, 'step_idx': 0, 'done': False, 'success': False,
                        'query_policy': policy, 'policy': None, 'client_type': 'fixture',
                        'obs': observation(0), 'os': SimpleNamespace(environ={}), 'sj_shadow': observer,
                        'np': np, 'action_space': SimpleNamespace(low=np.full(8, -1), high=np.full(8, 1), shape=(8,)),
                        'env': SimpleNamespace(step=step),
                        'torch': SimpleNamespace(from_numpy=lambda a: SimpleNamespace(unsqueeze=lambda axis: a)),
                        'extract_obs': lambda *args: observation(len(executed)), 'robot': None,
                        'scene_info': {'prompt': 'transport'}, 'episode_seed': 123, 'total_reward': 0.,
                        'ever_contacted': False, '_task_objs': [], '_target_obj': None,
                        'monitor': monitor, 'goal_checker': goal, 'goal_detail': {},
                        'success_first_step': None, 'consec_success': 0}
                    from runner_fixture import prepare_scope
                    prepare_scope(scope)
                    with patch('sys.stdout', new=io.StringIO()):
                        exec(code, scope)
                    final = stop_at or max_steps
                    observer.finish({'steps': final, 'status': 'completed', 'success': scope['success']})
                    self.assertEqual(requests, list(range(final+1)))
                    self.assertEqual(len(executed), final)
                    self.assertEqual(policy.call_count, (final+7)//8)
                    self.assertTrue(np.array_equal(chunk, original))
                    expected = np.full((final, 8), .2, dtype=np.float32)
                    expected[:, -1] = 1  # Original runner's gripper binarization.
                    np.testing.assert_array_equal(executed, expected)
                    self.assertEqual(scope['success'], stop_at is not None)
                    labels = read_jsonl(observer.directory/'classification-labels.jsonl')
                    self.assertEqual([r['step'] for r in labels if r['labels']['bad']], [3])


if __name__ == '__main__':
    unittest.main()
