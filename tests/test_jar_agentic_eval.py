import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from safetyjev.agentic_config import DEFAULTS, public_config

spec = importlib.util.spec_from_file_location('jar_eval', Path(__file__).resolve().parents[1]/'scripts/remote/jar-agentic-eval.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class JarEvaluationTests(unittest.TestCase):
    def test_full_and_pilot_have_unique_paired_cases(self):
        scenes = [f'task_{i:04d}/base' for i in range(26)]
        for pilot, count in ((False,234),(True,15)):
            plan = runner.cases(scenes,pilot)
            self.assertEqual(len(plan),count)
            self.assertEqual(len(set(plan)),count)
            self.assertEqual([c[2] for c in plan[:3]],['baseline','local','system1'])

    def test_local_ablation_disables_repairs_without_changing_original(self):
        config = runner.method_config(DEFAULTS,'system1')
        config['system1']['api_key']='private-test-secret'
        local = runner.method_config(config,'local')
        self.assertEqual(local['local_retries'],2)
        self.assertFalse(local['system1']['enabled'])
        self.assertEqual(local['repairs_per_decision'],0)
        self.assertEqual(local['system1']['max_calls_per_episode'],0)
        self.assertEqual(local['system1']['api_key'],'')
        self.assertEqual(config['system1']['api_key'],'private-test-secret')
        self.assertNotIn('private-test-secret',json.dumps(public_config(config)))

    def episode(self, root):
        ep = root/'episodes'/'one'
        runner.save(ep/'complete.json',{'status':'completed','monitor_valid':True})
        runner.save(ep/'maniguard_result.json',{'status':'completed','steps':1})
        choice = {'candidate_id':'earlier','start_step':0,'commands':[[1]*8]*8,
                  'policy_instruction':'original instruction','fallback':True}
        (ep/'selections.jsonl').write_text(json.dumps(choice)+'\n')
        executed = {'candidate_id':'earlier','step':1,'executed_action':[1]*8,
                    'policy_instruction':'original instruction'}
        (ep/'executed-candidates.jsonl').write_text(json.dumps(executed)+'\n')
        return ep, executed

    def test_audit_accepts_exact_fallback_and_rejects_wrong_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep, record = self.episode(root)
            self.assertEqual(runner.audit(root,'system1')['fallbacks'],1)
            record['policy_instruction']='wrong repair instruction'
            (ep/'executed-candidates.jsonl').write_text(json.dumps(record)+'\n')
            with self.assertRaises(RuntimeError):runner.audit(root,'system1')

    def test_audit_rejects_missing_actions_and_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep, _ = self.episode(root)
            (ep/'executed-candidates.jsonl').unlink()
            with self.assertRaises(RuntimeError):runner.audit(root,'local')
            runner.save(ep/'maniguard_result.json',{'status':'completed','steps':1,
                'safetyjev_agentic':{'infrastructure_failure':'no_valid_candidate'}})
            with self.assertRaises(RuntimeError):runner.audit(root,'baseline')

    def test_summary_keeps_raw_and_gated_violations_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            record = {'status':'completed','method':'baseline','seed':0,'fallbacks':0,'repairs':0,
                'repair_errors':0,'result':{'success':True,'ever_contacted':True,
                                           'counted_violation':False,'ltl_violated':True}}
            runner.save(out/'cases/a/record.json',record)
            runner.save(out/'cases/b/record.json',{'status':'failed','method':'local','seed':0})
            result = runner.summarize(out,{'cases':[1,2,3]})
            rates = result['methods']['baseline']['per_seed']['0']
            self.assertEqual(rates['safe_success'],1)
            self.assertEqual(rates['raw_violation'],1)
            self.assertEqual(rates['engaged_violation'],0)
            self.assertEqual(result['failed'],1)
            self.assertFalse(result['complete_comparison'])

    def test_private_config_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            runner.save(path,{'key':'test'},private=True)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)


if __name__=='__main__':unittest.main()
