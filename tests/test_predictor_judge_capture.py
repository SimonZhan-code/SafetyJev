import json,tempfile,unittest
from pathlib import Path
import numpy as np
from safetyjev.predictor_judge_capture import RolloutRecorder
from predictor_judge_fixture import episode_fixture
from safetyjev.predictor_judge_schema import validate_episode_record

class PredictorJudgeCaptureTests(unittest.TestCase):
    def test_copies_proposals_and_aligns_transition_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=episode_fixture(steps=2,violation_step=None)
            meta={k:e[k] for k in ['episode_id','group_id','action_dt_s','state_features','constraints']}
            r=RolloutRecorder(tmp,meta,chunk_size=1)
            images={c:np.zeros((8,8,3),dtype=np.uint8) for c in ['overview','wrist']}
            def state(t):return {'step':t,'robot_state':e['observations'][t]['robot_state'],'policy_state':[0.]*8,'images':images,'physical':{'object_pose':[1.,2.,3.]}}
            r.start_episode(state(0),e['oracle'][0])
            proposal=dict(e['proposals'][0]);proposal['raw_actions']=np.asarray(proposal['raw_actions']);r.record_proposal(proposal);proposal['raw_actions'][0,0]=999
            for t in range(2):
                r.before_action(e['execution'][t]);r.after_step(state(t+1));r.record_oracle(e['oracle'][t+1])
            r.finish_episode({'status':'completed','steps':2,'success':False})
            saved=json.loads(Path(tmp,'record.json').read_text());validate_episode_record(saved)
            self.assertEqual(saved['proposals'][0]['raw_actions'][0][0],0)
            self.assertEqual(len(saved['observations']),3);self.assertEqual(len(saved['execution']),2)
            self.assertTrue(Path(tmp,'states/0000000.npz').exists())
            self.assertEqual(json.loads(Path(tmp,'episode_status.json').read_text())['recording_status'],'complete')
    def test_failed_action_keeps_attempt_but_not_executed_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=episode_fixture(steps=1);meta={k:e[k] for k in ['episode_id','group_id','action_dt_s','state_features','constraints']};r=RolloutRecorder(tmp,meta)
            r.start_episode({'step':0,'robot_state':[0.]*16,'images':{c:np.zeros((8,8,3),dtype=np.uint8) for c in ['overview','wrist']},'physical':{}},e['oracle'][0])
            r.record_proposal(e['proposals'][0]);r.before_action(e['execution'][0]);r.finish_episode({'status':'crashed','steps':0})
            record=json.loads(Path(tmp,'record.json').read_text());self.assertEqual(record['execution'],[])
            self.assertEqual(len(Path(tmp,'attempts.jsonl').read_text().splitlines()),1)
            self.assertEqual(json.loads(Path(tmp,'episode_status.json').read_text())['recording_status'],'incomplete')
    def test_missing_monitor_marks_step_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=episode_fixture(steps=1);meta={k:e[k] for k in ['episode_id','group_id','action_dt_s','state_features','constraints']};r=RolloutRecorder(tmp,meta)
            def state(t):return {'step':t,'robot_state':[0.]*16,'images':{c:np.zeros((8,8,3),dtype=np.uint8) for c in ['overview','wrist']},'physical':{}}
            r.start_episode(state(0),e['oracle'][0]);r.record_proposal(e['proposals'][0]);r.before_action(e['execution'][0]);r.after_step(state(1));r.finish_episode({'status':'monitor_failed','steps':1})
            record=json.loads(Path(tmp,'record.json').read_text());self.assertFalse(record['oracle'][1]['valid'])
    def test_returned_env_step_survives_observation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=episode_fixture(steps=1);meta={k:e[k] for k in ['episode_id','group_id','action_dt_s','state_features','constraints']};r=RolloutRecorder(tmp,meta)
            r.start_episode({'step':0,'robot_state':[0.]*16,'images':{c:np.zeros((8,8,3),dtype=np.uint8) for c in ['overview','wrist']},'physical':{}},e['oracle'][0])
            r.record_proposal(e['proposals'][0]);r.before_action(e['execution'][0]);r.action_applied(0)
            r.finish_episode({'status':'crashed','steps':0})
            record=json.loads(Path(tmp,'record.json').read_text());self.assertEqual(len(record['execution']),1)
            validate_episode_record(record)
            from safetyjev.predictor_judge_data import build_predictor_judge_samples
            self.assertEqual(build_predictor_judge_samples(record)[0]['label_reason'],'censored')
