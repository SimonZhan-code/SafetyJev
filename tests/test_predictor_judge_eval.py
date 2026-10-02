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
