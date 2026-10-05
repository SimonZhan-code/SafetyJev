import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TrackingTests(unittest.TestCase):
    def test_disabled_and_nonzero_rank_do_not_import_sdk_or_write(self):
        from safetyjev.experiment_tracking import ExperimentTracker
        with tempfile.TemporaryDirectory() as folder:
            for env in ({}, {'SAFETYJEV_TRACKING': 'wandb', 'RANK': '1'}):
                with patch.dict(os.environ, env, clear=True), patch('importlib.import_module') as importer:
                    tracker = ExperimentTracker(Path(folder)/'run')
                    tracker.start({'seed': 42})
                    tracker.step({'step': 1, 'loss': .2})
                    tracker.finish('completed')
                    importer.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_resume_reuses_id_and_logs_attempt_without_secret_or_paths(self):
        from safetyjev.experiment_tracking import ExperimentTracker
        sdk = MagicMock()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            'SAFETYJEV_TRACKING': 'wandb', 'WANDB_PROJECT': 'test-project',
            'WANDB_ENTITY': 'test-team', 'WANDB_MODE': 'offline', 'WANDB_API_KEY': 'SECRET',
        }, clear=True), patch('importlib.import_module', return_value=sdk):
            root = Path(folder)
            config = {'seed': 42, 'model': {'model_id': 'vendor/model', 'token': 'SECRET'},
                      'data': {'package': '/private/data', 'batch_size': 8}, 'training': {'lr': .01}}
            first = ExperimentTracker(root)
            first.start(config)
            run_id = sdk.init.call_args.kwargs['id']
            self.assertNotIn('SECRET', json.dumps(sdk.init.call_args.kwargs['config']))
            self.assertNotIn('/private', json.dumps(sdk.init.call_args.kwargs['config']))
            first.step({'step': 1, 'loss': .2, 'checkpoint': '/private/model'})
            first.finish('paused')
            second = ExperimentTracker(root, resume=True)
            second.start(config)
            self.assertEqual(sdk.init.call_args.kwargs['id'], run_id)
            self.assertEqual(sdk.init.call_args.kwargs['resume'], 'allow')
            second.step({'step': 1, 'loss': .3})
            logged = sdk.init.return_value.log.call_args.args[0]
            self.assertEqual(logged['optimizer_step'], 1)
            self.assertEqual(logged['attempt'], 2)
            self.assertNotIn('step', sdk.init.return_value.log.call_args.kwargs)
            self.assertNotIn('SECRET', (root/'tracking.json').read_text())
            with patch.dict(os.environ, {'WANDB_PROJECT': 'different'}):
                with self.assertRaisesRegex(ValueError, 'destination'):
                    ExperimentTracker(root, resume=True).start(config)

    def test_local_model_reference_is_not_sent_to_tracker(self):
        from safetyjev.experiment_tracking import public_config
        result=public_config({'model':{'model_id':'/private/model-cache/base','revision':None}})
        self.assertNotIn('/private',json.dumps(result))
        self.assertEqual(result['model']['model_id'],'local-model')

    def test_logging_failure_does_not_fail_training_or_echo_secrets(self):
        from safetyjev.experiment_tracking import ExperimentTracker
        sdk = MagicMock()
        sdk.init.return_value.log.side_effect = RuntimeError('SECRET')
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            'SAFETYJEV_TRACKING': 'wandb', 'WANDB_PROJECT': 'test',
        }, clear=True), patch('importlib.import_module', return_value=sdk):
            tracker = ExperimentTracker(folder)
            tracker.start({})
            with self.assertWarnsRegex(RuntimeWarning, 'local logs') as caught:
                tracker.step({'step': 1, 'loss': 1.})
            self.assertNotIn('SECRET', str(caught.warning))
            tracker.step({'step': 2, 'loss': 1.})
            self.assertEqual(sdk.init.return_value.log.call_count, 1)
            tracker.finish('completed')
