import tempfile,unittest
from pathlib import Path
from safetyjev.predictor_judge_commands import capture_command
class PredictorJudgeCommandTests(unittest.TestCase):
    def test_capture_refuses_legacy_checkout_and_never_guesses_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):capture_command(tmp,'raw','p.json',['--seed','0'])
    def test_capture_forwards_user_settings_without_mutating_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'maniguard/eval';p.mkdir(parents=True);(p/'recording.py').write_text('RECORDING_API_VERSION = 1\n')
            args=['--seed','0'];cmd,env=capture_command(root,'raw','p.json',args)
            self.assertEqual(args,['--seed','0']);self.assertIn('safetyjev.predictor_judge_capture:create_observer',cmd)
            self.assertTrue(env['PYTHONPATH'].startswith(str(root)))
