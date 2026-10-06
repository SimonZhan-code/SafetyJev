import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetyjev.visual_runtime import predicate_label, image_request, validate_answers, report, VisualClassificationEpisode
from safetyjev.capture import ShadowEpisode
from safetyjev.io import write_json, append_jsonl

QUERIES = json.loads((Path(__file__).parents[1]/'configs/jar-visual-queries.json').read_text())['queries']


class VisualRuntimeTests(unittest.TestCase):
    def test_question_polarity_and_current_conjunction(self):
        ap={'jar_upright':True,'jar_dropped':False,'jar_closed':False,'jar_on_support':True}
        self.assertEqual([predicate_label(q,ap) for q in QUERIES],[0,0,0,1,0])
        ap.update(jar_on_support=False)
        self.assertEqual([predicate_label(q,ap) for q in QUERIES],[0,0,0,0,1])
        ap.update(jar_closed=True)
        self.assertEqual([predicate_label(q,ap) for q in QUERIES],[0,0,1,0,0])
        with self.assertRaises(ValueError):predicate_label(QUERIES[0],{})

    def test_request_contains_only_images_and_question_text(self):
        request=image_request({'overview':b'image1','wrist':b'image2'},QUERIES)
        self.assertEqual(set(request),{'images','questions'})
        self.assertTrue(all(set(q)=={'id','question'} for q in request['questions']))
        self.assertNotIn('yes_if',json.dumps(request))
        self.assertNotIn('all_of',json.dumps(request))

    def test_cumulative_violation_does_not_override_current_ap(self):
        observer=VisualClassificationEpisode.__new__(VisualClassificationEpisode)
        observer.ready=False;observer.invalid=False
        class Monitor:
            violated=True
            _ltl_log=[{'step':8,'ap':{'jar_upright':True,'jar_dropped':False,'jar_closed':True,'jar_on_support':True}}]
        with patch.object(ShadowEpisode,'after_step'):
            observer.after_step(8,Monitor(),{})
        self.assertEqual(predicate_label(QUERIES[4],observer.current_ap),0)

    def test_invalid_or_missing_predictions_are_rejected(self):
        answers={q['id']:{'raw_yes':.2,'calibrated_yes':.3,'logit':-1.} for q in QUERIES}
        validate_answers({'answers':answers},QUERIES)
        answers[QUERIES[0]['id']]['raw_yes']=float('nan')
        with self.assertRaises(ValueError):validate_answers({'answers':answers},QUERIES)
        with self.assertRaises(ValueError):validate_answers({'answers':{}},QUERIES)

    def test_report_counts_failures_and_distinguishes_raw_calibrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ep=root/'episode';ep.mkdir()
            write_json(ep/'episode.json',{'scene_name':'task_0000/base','visual_classifier':{'queries':[QUERIES[0]]}})
            write_json(ep/'complete.json',{'status':'completed','monitor_valid':True});write_json(ep/'maniguard_result.json',{'success':False})
            for step in (0,8):append_jsonl(ep/'classification-labels.jsonl',{'sample_id':str(step),'labels':{'jar_tilted':1}})
            append_jsonl(ep/'classification-predictions.jsonl',{'sample_id':'0','latency_s':1.,'error':None,
                'answers':{'jar_tilted':{'raw_yes':.4,'calibrated_yes':.6,'logit':-.4}}})
            result=report(root,root/'report.json')
            self.assertEqual(result['coverage'],.5)
            self.assertEqual(result['failed_or_missing_classifications'],1)
            self.assertEqual(result['raw']['micro']['fn'],1)
            self.assertEqual(result['calibrated']['micro']['tp'],1)


if __name__=='__main__':unittest.main()
