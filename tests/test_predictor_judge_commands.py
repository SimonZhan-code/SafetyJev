import tempfile,unittest
from pathlib import Path
from safetyjev.predictor_judge_commands import capture_command
class PredictorJudgeCommandTests(unittest.TestCase):
    def test_build_cli_selects_unsafe_train_and_keeps_safe_heldout(self):
        import json
        from PIL import Image
        from predictor_judge_fixture import episode_fixture
        from safetyjev.predictor_judge_commands import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assignments={}
            for eid,split,violation in [('u','train',6),('s','train',None),('v','validation',None),('t','test',6)]:
                d=root/'raw'/eid;d.mkdir(parents=True)
                e=episode_fixture(violation_step=violation);e.update(episode_id=eid,group_id='jar/'+eid)
                assignments[e['group_id']]=split
                for o in e['observations']:
                    for name in o['images'].values():
                        p=d/name;p.parent.mkdir(exist_ok=True);Image.new('RGB',(4,4)).save(p)
                (d/'record.json').write_text(json.dumps(e))
            manifest=root/'splits.json';manifest.write_text(json.dumps(assignments))
            for mode,expected in [('unsafe_only',{'u','v','t'}),('unsafe_plus_safe',{'u','s','v','t'})]:
                output=root/mode
                main(['build','--episodes',str(root/'raw'),'--output',str(output),
                      '--split-manifest',str(manifest),'--train-episodes',mode])
                inventory=[json.loads(s) for s in (output/'episode_inventory.jsonl').read_text().splitlines()]
                self.assertEqual({r['episode_id'] for r in inventory if r['selected']},expected)
                self.assertTrue((root/'raw/s/record.json').is_file())

    def test_capture_refuses_legacy_checkout_and_never_guesses_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):capture_command(tmp,'raw','p.json',['--seed','0'])
    def test_capture_forwards_user_settings_without_mutating_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'maniguard/eval';p.mkdir(parents=True);(p/'recording.py').write_text('RECORDING_API_VERSION = 1\n')
            args=['--seed','0'];cmd,env=capture_command(root,'raw','p.json',args)
            self.assertEqual(args,['--seed','0']);self.assertIn('safetyjev.predictor_judge_capture:create_observer',cmd)
            self.assertTrue(env['PYTHONPATH'].startswith(str(root)))
    def test_build_discovers_nested_campaign_episodes_once(self):
        from unittest.mock import patch
        from safetyjev.predictor_judge_commands import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'run/raw/id';episode.mkdir(parents=True)
            for name in ['record.json','episode.json','episode_status.json']:(episode/name).write_text('{}')
            with patch('safetyjev.predictor_judge_commands.build_package') as build:
                main(['build','--episodes',str(root),'--output',str(root/'package')])
            self.assertEqual(build.call_args.args[0],[episode])
