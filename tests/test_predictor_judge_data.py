import copy
import unittest
from predictor_judge_fixture import episode_fixture
from safetyjev.predictor_judge_data import build_predictor_judge_samples, label_predictor_judge_sample, split_groups

class PredictorJudgeDataTests(unittest.TestCase):
    def test_windows_follow_committed_suffix_and_adjacent_history(self):
        rows=build_predictor_judge_samples(episode_fixture(violation_step=None))
        self.assertEqual(len(rows),16)
        self.assertEqual([r['valid_steps'] for r in rows],list(range(8,0,-1))*2)
        self.assertEqual(rows[0]['history_steps'],[None,None,0])
        self.assertEqual(rows[3]['history_steps'],[1,2,3])
        self.assertEqual(rows[3]['end_step'],8)
        self.assertEqual(rows[8]['proposal_id'],'p8')
        self.assertTrue(all(r['target']==[1.,0.] for r in rows))
    def test_positive_censor_and_already_violated_are_distinct(self):
        rows=build_predictor_judge_samples(episode_fixture(steps=5,violation_step=3))
        self.assertEqual(rows[0]['target'],[0.,1.])
        self.assertEqual(rows[3]['label_reason'],'already_violated')
        rows=build_predictor_judge_samples(episode_fixture(steps=5,violation_step=None))
        self.assertTrue(all(r['label_reason']=='censored' for r in rows))
    def test_oracle_gap_does_not_bridge_to_later_positive(self):
        e=episode_fixture();e['oracle'][4]['valid']=False
        rows=build_predictor_judge_samples(e)
        self.assertIsNone(rows[0]['target']);self.assertEqual(rows[5]['target'],[0.,1.])
    def test_replacement_cannot_label_original_candidate(self):
        e=episode_fixture(violation_step=6);rows=build_predictor_judge_samples(e)
        p=copy.deepcopy(e['proposals'][0]);p.update(proposal_id='replacement',start_step=4)
        e['proposals'].append(p)
        for t in range(4,8):e['execution'][t].update(proposal_id='replacement',offset=t-4,command=p['planned_commands'][t-4])
        result=label_predictor_judge_sample(rows[0],e)
        self.assertIsNone(result['target']);self.assertEqual(result['label_reason'],'action_replaced')
    def test_temporal_samples_keep_monitor_supervision_without_oracle_input(self):
        e=episode_fixture();e['constraints'][0]['requires_history']=True
        rows=build_predictor_judge_samples(e)
        self.assertEqual(len(rows),16)
        self.assertEqual(rows[0]['target'],[0.,1.])
        self.assertEqual(rows[0]['constraint_context'],{'source':'none','text':''})
        self.assertTrue(rows[0]['requires_history'])
        self.assertEqual(rows[6]['label_reason'],'already_violated')
    def test_splits_keep_entire_base_task_together(self):
        rows=[{'group_id':f'jar/{g}','id':f'{g}-{i}'} for g in range(20) for i in range(3)]
        split=split_groups(rows,seed=42)
        self.assertEqual(set(split.values()),{'train','validation','test'})
        self.assertEqual(split,split_groups(rows[::-1],seed=42))
    def test_later_invalid_proposal_keeps_earlier_positive_samples(self):
        e=episode_fixture(steps=11,violation_step=2);e['result']={'status':'numerical_failed'}
        e['proposals'][1]['planned_commands'][3]=[None]*8;e['proposals'][1]['raw_actions'][3]=[None]*8
        rows=build_predictor_judge_samples(e)
        self.assertEqual(rows[0]['target'],[0.,1.])
        self.assertEqual(rows[8]['label_reason'],'invalid_actions');self.assertIsNone(rows[8]['target'])
        self.assertEqual(rows[8]['end_step'],16)
