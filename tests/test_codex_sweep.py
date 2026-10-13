import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('codex_sweep',Path(__file__).parents[1]/'scripts/remote/codex-robot-smoke.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class CodexSweepTests(unittest.TestCase):
    def test_grid_is_exact_and_smoke_is_separate(self):
        grid=queue.case_grid('jar-base')
        self.assertEqual(len(grid),78)
        self.assertEqual(set(grid),{(f'task_{i:04d}/base',s) for i in range(26) for s in range(3)})
        self.assertEqual(queue.case_grid('smoke'),[('task_0000/base',0),('task_0005/base',0)])

    def test_sweep_requires_matching_completed_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cfg={'max_steps':2000,'model':'gpt-6-astra'}
            config_path=root/'full.json'; hashes={'runtime.py':'a','semantics.json':'b',str(config_path.resolve()):'c'}
            plan={'suite':'smoke','cases':[[s,n] for s,n in queue.case_grid('smoke')],
                  'config':dict(cfg,max_steps=256),'maniguard':{'revision':'dev-pin'},
                  'source_sha256':{'runtime.py':'a','semantics.json':'b'}}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'progress.json').write_text(json.dumps({'status':'completed','completed':2,'planned':2}))
            for scene in queue.CASES:
                ep=root/scene.replace('/','-')/'episodes'/'one';ep.mkdir(parents=True)
                (ep/'complete.json').write_text('{}')
            with patch.object(queue.jar,'source_provenance',return_value={'revision':'dev-pin'}), patch.object(queue,'audit',return_value={'status':'completed'}):
                queue.verify_smoke(root,cfg,hashes,config_path)
                for key in ('runtime.py','semantics.json'):
                    with self.assertRaisesRegex(ValueError,'runtime differs'):
                        queue.verify_smoke(root,cfg,dict(hashes,**{key:'changed'}),config_path)
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    queue.verify_smoke(root,dict(cfg,model='other'),hashes,config_path)
                (root/'progress.json').write_text(json.dumps({'status':'running','completed':1,'planned':2}))
                with self.assertRaisesRegex(ValueError,'Both smoke'):
                    queue.verify_smoke(root,cfg,hashes,config_path)
        with self.assertRaisesRegex(ValueError,'smoke-evidence'):
            queue.verify_smoke(None,{}, {},Path('/tmp/cfg'))

    def test_predecessor_excludes_only_audited_completed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'plan.json').write_text(json.dumps({'suite':'jar-base','max_steps':2000,'maniguard':{'revision':'dev'}}))
            for seed,status in ((0,'completed'),(1,'running'),(2,'failed')):
                case=root/('task_0000-base-seed'+str(seed));case.mkdir()
                (case/'record.json').write_text(json.dumps({'status':status}))
                ep=case/'episodes'/'one';ep.mkdir(parents=True);(ep/'complete.json').write_text('{}')
            with patch.object(queue.jar,'source_provenance',return_value={'revision':'dev'}), patch.object(queue,'audit',return_value={'status':'completed'}) as audit:
                rows=queue.completed_predecessor(root)
                self.assertEqual([(r['scene'],r['seed']) for r in rows],[('task_0000/base',0)])
                self.assertEqual(audit.call_count,1)
                self.assertEqual(len(rows[0]['record_sha256']),64)
                audit.return_value={'status':'infrastructure_failure'}
                with self.assertRaisesRegex(ValueError,'audit failed'):queue.completed_predecessor(root)
            with patch.object(queue.jar,'source_provenance',return_value={'revision':'other'}):
                with self.assertRaisesRegex(ValueError,'Incompatible'):queue.completed_predecessor(root)

    def test_failures_do_not_become_task_outcomes(self):
        good={'status':'completed','task_success':True,'safe_success':True,'raw_violation':True,
              'result':{'counted_violation':False,'ever_contacted':True,'ever_grasped':True},
              'predicted_unsafe_selected':2,'higher_score_available':1,'invalid_attempts':0,
              'proposal_sources':{'pi05':3,'codex':1}}
        result=queue.summarize([good,{'status':'failed'},{'status':'running'}],78)
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['failed'],1)
        self.assertEqual(result['task_success'],1)
        self.assertEqual(result['success_without_raw_violation'],0)
        self.assertEqual(result['proposal_sources'],{'pi05':3,'codex':1})


if __name__=='__main__':unittest.main()
