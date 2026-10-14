import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('codex_sweep',Path(__file__).parents[1]/'scripts/remote/codex-robot-smoke.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class CodexSweepTests(unittest.TestCase):
    def test_grid_is_exact_and_smoke_is_separate(self):
        self.assertEqual(queue.case_grid('timing'),[('task_0000/base',1)])
        self.assertEqual(queue.case_grid('single-smoke'),[('task_0000/base',1)])
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

    def test_explicit_skills_single_admission_keeps_runtime_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={'max_steps':2000,'tool_sequence':'skills'}
            config_path=root/'full.json';launcher=str(Path(queue.__file__).resolve())
            hashes={'runtime.py':'a',launcher:'new',str(config_path.resolve()):'c'}
            plan={'suite':'single-smoke','cases':[['task_0000/base',1]],
                  'config':dict(cfg,max_steps=256),'maniguard':{'revision':'dev'},
                  'source_sha256':{'runtime.py':'a',launcher:queue.SINGLE_SMOKE_LAUNCHER_SHA256}}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'progress.json').write_text(json.dumps({'status':'completed','completed':1,'planned':1}))
            ep=root/'task_0000-base-seed1/episodes/one';ep.mkdir(parents=True);(ep/'complete.json').write_text('{}')
            with patch.object(queue.jar,'source_provenance',return_value={'revision':'dev'}), patch.object(queue,'audit',return_value={'status':'completed'}):
                with self.assertRaises(ValueError):queue.verify_smoke(root,cfg,hashes,config_path)
                queue.verify_smoke(root,cfg,hashes,config_path,True)
                with self.assertRaisesRegex(ValueError,'runtime differs'):
                    queue.verify_smoke(root,cfg,dict(hashes,**{'runtime.py':'changed'}),config_path,True)
                plan['source_sha256'][launcher]='unknown';(root/'plan.json').write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError,'runtime differs'):queue.verify_smoke(root,cfg,hashes,config_path,True)
                with self.assertRaisesRegex(ValueError,'only to bounded skills'):queue.verify_smoke(root,dict(cfg,tool_sequence='paired'),hashes,config_path,True)

    def test_queue_only_smoke_admission_rejects_other_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={'max_steps':2000,'tool_sequence':'skills'}
            config_path=root/'full.json';launcher=str(Path(queue.__file__).resolve())
            hashes={'runtime.py':'same',launcher:'updated'}
            plan={'suite':'smoke','cases':[[s,n] for s,n in queue.case_grid('smoke')],
                  'config':dict(cfg,max_steps=256),'maniguard':{'revision':'dev'},
                  'source_sha256':{'runtime.py':'same',launcher:queue.TIMEOUT_QUEUE_SHA256}}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'progress.json').write_text(json.dumps({'status':'completed','completed':2,'planned':2}))
            for scene in queue.CASES:
                ep=root/scene.replace('/','-')/'episodes'/'one';ep.mkdir(parents=True);(ep/'complete.json').write_text('{}')
            with patch.object(queue.jar,'source_provenance',return_value={'revision':'dev'}),patch.object(queue,'audit',return_value={'status':'completed'}):
                with self.assertRaisesRegex(ValueError,'runtime differs'):queue.verify_smoke(root,cfg,hashes,config_path)
                queue.verify_smoke(root,cfg,hashes,config_path,queue_update=True)
                with self.assertRaisesRegex(ValueError,'runtime differs'):
                    queue.verify_smoke(root,cfg,dict(hashes,**{'runtime.py':'changed'}),config_path,queue_update=True)
                plan['source_sha256'][launcher]='unknown';(root/'plan.json').write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError,'runtime differs'):
                    queue.verify_smoke(root,cfg,hashes,config_path,queue_update=True)

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


class TimeoutContinuationTests(unittest.TestCase):
    def report(self, reason='skill_or_planner_timeout'):
        return {'status':'infrastructure_failure','executed_steps':384,
                'result':{'safetyjev_codex':{'infrastructure_failure':reason}}}

    def test_audited_timeout_is_recorded_even_with_nonzero_capture_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);ep=case/'episodes/one';ep.mkdir(parents=True)
            (ep/'complete.json').write_text('{}')
            with patch.object(queue,'audit',return_value=self.report()):
                r=queue.terminal_report(case,'task_0001/base',1,420,1,True)
                self.assertEqual(r['status'],'timeout')
                self.assertEqual(r['queue_action'],'continue_next_case')
                self.assertEqual(r['executed_steps'],384)
                (case/'record.json').write_text(json.dumps(r))
                self.assertTrue(queue.skip_terminal_case(case,True))
                with self.assertRaisesRegex(RuntimeError,'original queue policy'):
                    queue.skip_terminal_case(case,False)
            for reason in ('predictor_failed','codex_ended_without_execution','skill_cancelled'):
                with patch.object(queue,'audit',return_value=self.report(reason)):
                    self.assertEqual(queue.terminal_report(case,'task_0001/base',1,420,1,True)['status'],'infrastructure_failure')
            with patch.object(queue,'audit',return_value=self.report()):
                self.assertEqual(queue.terminal_report(case,'task_0001/base',1,420,1,False)['status'],'infrastructure_failure')
            with patch.object(queue,'audit',side_effect=ValueError('command mismatch')):
                with self.assertRaisesRegex(ValueError,'command mismatch'):
                    queue.terminal_report(case,'task_0001/base',1,420,1,True)
            (ep/'complete.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'terminal episode'):
                queue.terminal_report(case,'task_0001/base',1,420,1,True)

    def test_timeout_does_not_inflate_task_success_or_completed_count(self):
        timed={'status':'timeout','task_success':True,'safe_success':True}
        self.assertEqual(queue.summarize([timed],1)['task_success'],0)
        self.assertEqual(queue.final_progress([timed],1),
            {'status':'completed_with_timeouts','planned':1,'completed':0,'timeouts':1,'attempted':1,'failed':1})
        self.assertEqual(queue.final_progress([],0),{'status':'completed','completed':0,'planned':0})
        with self.assertRaisesRegex(RuntimeError,'unprocessed'):
            queue.final_progress([{'status':'failed'}],1)

    def test_real_queue_advances_after_timeout_and_restart_skips_both(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory);out=root/'out';mg=root/'mg';bench=root/'bench'
            cfg=root/'config.json'
            cfg.write_text(json.dumps({'tool_sequence':'skills','proposal_limit':2,
                'decision_timeout_s':300,'max_steps':2000}))
            for relative in ('scripts/remote/jar-agentic-eval.py','scripts/remote/serve-pi05.sh',
                    'scripts/remote/warm-jar-policy.py','configs/jar-isaac51-provenance.json',
                    'configs/predictor-semantic-definitions.json','mg/configs/eval/jar_transport_joint.yaml',
                    'bench/task_0000/base/scene.json'):
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            good={'status':'completed','task_success':False,'safe_success':False,'raw_violation':False,
                'result':{},'predicted_unsafe_selected':0,'higher_score_available':0,
                'invalid_attempts':0,'proposal_sources':{},'executed_steps':2000}
            def audited(ep):
                return self.report() if 'seed0' in str(ep) else dict(good)
            captures=[]
            def run(command, **kwargs):
                if 'capture' in command:
                    ep=Path(command[command.index('--output')+1])/'one';ep.mkdir(parents=True)
                    (ep/'complete.json').write_text('{}');captures.append(str(ep))
                    return SimpleNamespace(returncode=1 if 'seed0' in str(ep) else 0)
                return SimpleNamespace(returncode=0)
            original_open=Path.open
            def redirected_open(p,*args,**kwargs):
                if str(p)=='/workspace/safetyjev-jar-agentic.lock':p=root/'queue.lock'
                return original_open(p,*args,**kwargs)
            for obj,name,value in ((queue,'ROOT',root),(queue.jar,'MANIGUARD',str(mg)),
                    (queue.jar,'BENCH',bench)):
                stack.enter_context(patch.object(obj,name,value))
            for obj,name,value in ((queue,'case_grid',[('task_0000/base',0),('task_0000/base',1)]),
                    (queue,'preflight',{}),(queue.jar,'source_provenance',{}),
                    (queue.jar,'verify_sources',None),(queue.jar,'verify_latest_dev',None),
                    (queue,'verify_smoke',None),(queue,'no_overlap',None),(queue,'create_services',None),
                    (queue.jar,'wait_for',True)):
                stack.enter_context(patch.object(obj,name,return_value=value))
            services=stack.enter_context(patch.object(queue.jar,'service'))
            stack.enter_context(patch.object(queue,'audit',side_effect=audited))
            stack.enter_context(patch.object(queue.subprocess,'run',side_effect=run))
            stack.enter_context(patch.object(Path,'open',redirected_open))
            stack.enter_context(patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
            stack.enter_context(patch.object(queue.sys,'argv',['queue','--suite','jar-base','--config',str(cfg),
                '--output',str(out),'--smoke-evidence',str(root/'smoke'),'--continue-on-timeout']))
            queue.main()
            self.assertEqual(len(captures),2)
            progress=json.loads((out/'progress.json').read_text())
            self.assertEqual(progress['status'],'completed_with_timeouts')
            self.assertEqual((progress['completed'],progress['timeouts']),(1,1))
            # Timeout cleanup stops both services and restarts them for the next case.
            self.assertEqual(services.call_args_list[2].args,('stop',queue.SERVICES[1]))
            self.assertEqual(services.call_args_list[3].args,('stop',queue.SERVICES[0]))
            self.assertEqual(services.call_args_list[4].args,('start',queue.SERVICES[0]))
            queue.main()
            self.assertEqual(len(captures),2)

    def test_300_second_watchdog_has_longer_mcp_envelope(self):
        from safetyjev.codex_config import validate_config
        from safetyjev.codex_dispatch_audit import dispatch_config
        cfg=validate_config({'tool_sequence':'skills','proposal_limit':2,'decision_timeout_s':300})
        actual=dispatch_config('/socket','/catalog','skills',cfg['decision_timeout_s'])
        self.assertEqual(actual['mcp_servers']['robot']['tool_timeout_sec'],330)
        self.assertEqual(actual['mcp_servers']['robot']['enabled_tools'],['observe','run_skill'])
        with self.assertRaises(ValueError):
            validate_config({'tool_sequence':'skills','proposal_limit':2,'decision_timeout_s':301})


class InitializationContinuationTests(unittest.TestCase):
    def fixture(self, root):
        case=root/'task_0002-base-seed0';case.mkdir()
        out=root/'mg/outputs/eval_logs/jar_transport_joint/run';out.mkdir(parents=True)
        result={'status':'initialization_failed','scene_name':'task_0002/base','seed':0,'steps':0,
                'success':None,'ltl_violated':None,'counted_violation':None,'safety_evaluated':False,
                'rollout_diagnostics':{'termination_reason':'initialization_error','phase':'initial_safety'}}
        (out/'results.jsonl').write_text(json.dumps(result)+'\n')
        cfg={'scenes':['task_0002/base'],'seed':0,'tag':'codex-harness-'+case.name,'max_steps':2000}
        (out/'eval_config.json').write_text(json.dumps(cfg))
        (case/'capture.log').write_text('Results: outputs/eval_logs/jar_transport_joint/run/results.jsonl\n')
        return case,out,result

    def test_verified_zero_action_failure_continues_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case,out,result=self.fixture(root)
            with patch.object(queue.jar,'MANIGUARD',str(root/'mg')):
                with self.assertRaisesRegex(RuntimeError,'terminal episode'):
                    queue.terminal_report(case,'task_0002/base',0,40,0,True)
                r=queue.terminal_report(case,'task_0002/base',0,40,0,True,True)
                self.assertEqual(r['status'],'initialization_failed')
                self.assertIsNone(r['task_success']);self.assertFalse(r['execution_audit_passed'])
                original={'status':'failed','scene':'task_0002/base','seed':0}
                (case/'record.json').write_text(json.dumps(original))
                self.assertTrue(queue.skip_terminal_case(case,True,True))
                self.assertEqual(json.loads((case/'record-before-initialization-classification.json').read_text()),original)
                self.assertTrue(queue.skip_terminal_case(case,True,True))
                final=queue.final_progress([r],1)
                self.assertEqual(final['status'],'completed_with_failures')
                self.assertEqual((final['completed'],final['initialization_failures'],final['timeouts']),(0,1,0))
                result['error']='modified evidence';(out/'results.jsonl').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'evidence changed'):
                    queue.skip_terminal_case(case,True,True)

    def test_mismatches_or_policy_evidence_never_become_initialization_skips(self):
        for field,value in (('steps',1),('steps',False),('scene_name','task_0003/base'),('seed',1),('success',False),('safety_evaluated',True)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);case,out,result=self.fixture(root);result[field]=value
                (out/'results.jsonl').write_text(json.dumps(result))
                with patch.object(queue.jar,'MANIGUARD',str(root/'mg')),self.assertRaises(ValueError):
                    queue.initialization_report(case,'task_0002/base',0,40,0)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case,out,result=self.fixture(root)
            ep=case/'episodes/one';ep.mkdir(parents=True);(ep/'harness-events.jsonl').write_text('')
            with patch.object(queue.jar,'MANIGUARD',str(root/'mg')),self.assertRaisesRegex(ValueError,'policy/command'):
                queue.initialization_report(case,'task_0002/base',0,40,0)

    def test_queue_amendment_cannot_change_models_scenes_or_controller(self):
        import copy
        launcher=str(Path(queue.__file__).resolve())
        old={'config':{'decision_timeout_s':300},'source_sha256':{launcher:queue.TIMEOUT_QUEUE_SHA256,'controller.py':'same'},'cases':[['task_0002/base',0]]}
        new=copy.deepcopy(old);new['source_sha256'][launcher]='updated';new['initialization_failure_policy']='record_and_continue'
        queue.validate_queue_update(old,new)
        for key,value in (('config',{'decision_timeout_s':600}),('cases',[]),('source_sha256',{launcher:'updated','controller.py':'changed'})):
            bad=copy.deepcopy(new);bad[key]=value
            with self.assertRaisesRegex(ValueError,'more than'):
                queue.validate_queue_update(old,bad)


if __name__=='__main__':unittest.main()
