import unittest
import torch
from safetyjev.predictor_judge_eval import evaluate_predictor_judge

class FixedModel:
    def eval(self):return self
    def __call__(self,**kw):return kw['logits']
class PredictorJudgeEvalTests(unittest.TestCase):
    def test_metrics_are_partitioned_by_constraint_and_real_horizon(self):
        batch={'inputs':{'logits':torch.tensor([[0.,3.],[0.,-3.],[0.,3.]])},
               'targets':torch.tensor([[0.,1.],[1.,0.],[1.,0.]]),
               'sample_ids':['a','b','c'],'query_ids':['tilt','tilt','drop'],'valid_steps':[8,1,1]}
        result=evaluate_predictor_judge(FixedModel(),[batch])
        self.assertEqual(result['micro']['tp'],1);self.assertEqual(result['micro']['fp'],1)
        self.assertEqual(result['by_valid_steps']['8']['n'],1)
        self.assertEqual(result['by_constraint']['tilt']['n'],2)
        self.assertGreater(result['nll'],0)
    def test_invalid_scores_fail_and_empty_evaluation_is_explicit(self):
        with self.assertRaises(ValueError):evaluate_predictor_judge(FixedModel(),[])
    def test_events_are_deduplicated_and_safe_episode_false_alarms_counted(self):
        from safetyjev.predictor_judge_eval import summarize_predictions
        rows=[]
        for i,(y,p,eid,start,event,safety) in enumerate([(1,.8,'u',1,5,'unsafe'),(1,.9,'u',3,5,'unsafe'),(1,.1,'v',2,6,'unsafe'),(0,.8,'s',1,None,'safe'),(0,.1,'s',2,None,'safe')]):
            rows.append({'id':str(i),'label':y,'score':p,'nll':1.,'constraint_id':'tilt','valid_steps':8,
                'metadata':{'episode_id':eid,'family':'jar','episode_safety':safety,'start_step':start,'first_violation_step':event}})
        report=summarize_predictions(rows)
        self.assertEqual(report['events'],{'eligible':2,'detected':1,'recall':.5,'mean_first_detected_lead_steps':4.})
        self.assertEqual(report['safe_source_episodes']['n'],1)
        self.assertEqual(report['safe_source_episodes']['any_false_alarm_rate'],1.)
        self.assertEqual(report['by_family_constraint']['jar/tilt']['n'],5)
        with self.assertRaisesRegex(ValueError,'Duplicate'):summarize_predictions(rows+rows[:1])

    def test_duration_includes_loader_separately_from_model_call(self):
        from unittest.mock import patch
        batch=dict(inputs={'logits':torch.tensor([[0.,1.]])},targets=torch.tensor([[0.,1.]]),
                   sample_ids=['x'],query_ids=['tilt'],valid_steps=[8])
        with patch('safetyjev.predictor_judge_eval.time.perf_counter',side_effect=[0.,1.,4.,8.]):
            report=evaluate_predictor_judge(FixedModel(),[batch])
        self.assertEqual(report['model_batch_seconds'],3.)
        self.assertEqual(report['evaluation_seconds'],8.)
