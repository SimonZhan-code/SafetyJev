import ast
import base64
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from safetyjev.agentic import AgenticEpisode, AgenticInfrastructureError
from safetyjev.agentic_config import DEFAULTS, IDENTITY, public_config, validate_config
from safetyjev.agentic_inputs import ChunkHistory, bind_queries, controller_commands, robot_state
from safetyjev.agentic_predictor import NativePredictor
from safetyjev.agentic_server import decode_inputs
from safetyjev.capture import ShadowEpisode
from safetyjev.cli import main
from safetyjev.io import read_jsonl
from safetyjev.maniguard import instrument, verify_sources
from safetyjev.planner import OpenRouterPlanner, PlannerError


class AgenticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'frames').mkdir()
        self.images = {'overview':'frames/overview.png','wrist':'frames/wrist.png'}
        for file in self.images.values():
            Image.new('RGB',(16,16)).save(self.root/file)
        g = self.g = AgenticEpisode.__new__(AgenticEpisode)
        g.config = validate_config({})
        g.directory = self.root
        g.episode_id = 'fixture'
        g.cfg = SimpleNamespace(action_frequency=20)
        g.native_history = ChunkHistory()
        g.native_history.append(0, [0.]*16, self.images)
        g.executed_history = []
        g.questions = [{'id':'a','question':'Will constraint A be violated?'},
                       {'id':'b','question':'Will constraint B be violated?'}]
        g.queries = list(g.questions)
        g.native_predictor = Mock()
        g.planner = Mock()
        g.planner.plan.return_value = {'vla_instruction':'Close the lid before lifting.'}
        g.planner_calls = g.fallbacks = g.decisions = g.candidate_attempts = 0
        g.failure_reason = None
        g._selected = None
        self.obs = {'states':np.zeros(8), 'task_descriptions':'Move the jar', 'episode_seed':123,
                    'overview_image':np.zeros((16,16,3),dtype=np.uint8),
                    'wrist_images':np.zeros((16,16,3),dtype=np.uint8)}
        self.bounds = SimpleNamespace(low=np.full(8,-1),high=np.full(8,1),shape=(8,))
        self.sequence = iter([self.chunk(v) for v in (.2,.3,.4,.5,.6,.7,.8)])
        self.regenerate = Mock(side_effect=lambda obs:next(self.sequence))

    def chunk(self, value=.1):
        return np.full((8,8),value,dtype=np.float32)

    def run_selection(self, answers):
        self.g.native_predictor.score.side_effect = answers
        return self.g.select_chunk(0,self.obs,self.chunk(),self.regenerate,self.bounds,8)

    def rows(self,name):
        return read_jsonl(self.root/name)

    def test_first_acceptance_needs_no_repair(self):
        self.run_selection([{'a':False,'b':False}])
        self.regenerate.assert_not_called(); self.g.planner.plan.assert_not_called()
        self.assertEqual(self.g.fallbacks,0)
        self.assertEqual(self.rows('selections.jsonl')[0]['satisfied_constraints'],2)

    def test_local_retry_before_system1_and_binary_only_feedback(self):
        self.run_selection([{'a':True,'b':False},{'a':False,'b':False}])
        self.assertEqual(self.regenerate.call_count,1)
        self.g.planner.plan.assert_not_called()
        self.assertEqual(self.regenerate.call_args.args[0]['task_descriptions'],'Move the jar')

    def test_six_candidate_fallback_can_select_pre_repair_candidate(self):
        chosen=self.run_selection([{'a':True,'b':True},{'a':False,'b':True},
            {'a':True,'b':True},{'a':True,'b':True},{'a':True,'b':False},{'a':True,'b':True}])
        self.assertEqual(self.g.candidate_attempts,6)
        self.assertEqual(self.regenerate.call_count,5)
        self.assertEqual(self.g.planner_calls,1)
        self.assertAlmostEqual(chosen[0,0],.2)
        selected=self.rows('selections.jsonl')[0]
        self.assertTrue(selected['fallback'])
        self.assertEqual(selected['policy_instruction'],'Move the jar')
        self.assertTrue(selected['candidate_id'].endswith('candidate-1'))
        self.assertEqual(self.regenerate.call_args.args[0]['task_descriptions'],'Close the lid before lifting.')
        ctx=self.g.planner.plan.call_args.args[0]
        self.assertNotIn('score',json.dumps(ctx['guard_feedback']))
        self.assertEqual(len(ctx['candidate_history']),3)
        self.assertEqual(ctx['executed_history'],[])
        self.assertEqual(self.obs['task_descriptions'],'Move the jar')

    def test_duplicates_consume_attempts_but_cache_complete_answers(self):
        self.regenerate.side_effect=None; self.regenerate.return_value=self.chunk()
        self.run_selection([{'a':True,'b':False}])
        self.assertEqual(self.g.candidate_attempts,6)
        self.assertEqual(self.g.native_predictor.score.call_count,1)
        self.assertEqual(sum(r['duplicate'] for r in self.rows('candidate-decisions.jsonl')),5)
        self.assertTrue(self.rows('selections.jsonl')[0]['candidate_id'].endswith('candidate-0'))

    def test_episode_budget_skips_system1_but_keeps_local_search(self):
        self.g.planner_calls=3
        self.run_selection([{'a':True,'b':False}]*3)
        self.g.planner.plan.assert_not_called()
        self.assertEqual(self.regenerate.call_count,2)
        self.assertEqual(self.g.fallbacks,1)

    def test_planner_error_consumes_call_and_selects_existing_candidate(self):
        self.g.planner.plan.side_effect=PlannerError('offline')
        self.run_selection([{'a':True,'b':False}]*3)
        self.assertEqual(self.g.planner_calls,1)
        self.assertEqual(self.g.candidate_attempts,3)
        self.assertEqual(self.rows('selections.jsonl')[0]['reason'],'planner_error')

    def test_unknown_answers_never_count_as_safe_and_all_unknown_is_failure(self):
        self.g.config['repairs_per_decision']=0
        selected=self.run_selection([{'a':False},{'a':True,'b':False},TimeoutError('private')])
        self.assertAlmostEqual(selected[0,0],.2)
        self.assertEqual(self.rows('candidate-decisions.jsonl')[0]['satisfied_constraints'],None)
        self.assertNotIn('private', (self.root/'candidate-decisions.jsonl').read_text())
        with self.assertRaises(AgenticInfrastructureError):
            self.run_selection([{}, {'a':False}, {'a':False,'b':None}])
        self.assertEqual(self.g.failure_reason,'no_valid_candidate')
        self.assertIsNone(self.g._selected)

    def test_invalid_chunk_is_ineligible_even_when_previous_candidate_passed(self):
        self.g.config['repairs_per_decision']=0
        self.g.native_predictor.score.return_value={'a':False,'b':False}
        selected=self.g.select_chunk(0,self.obs,np.zeros((7,8)),self.regenerate,self.bounds,8)
        self.assertAlmostEqual(selected[0,0],.2)
        self.assertEqual(self.g.native_predictor.score.call_count,1)
        self.assertFalse(self.rows('candidate-decisions.jsonl')[0]['valid'])

    def test_history_masks_and_native_state_schema(self):
        h=self.g.native_history
        inputs=h.inputs(self.chunk(),.05,self.root)
        self.assertEqual(inputs['history_mask'],[False]*7+[True])
        self.assertEqual(inputs['history_action_mask'],[False]*8)
        self.assertEqual(decode_inputs(inputs)['observations']['overview'].shape,(8,3,16,16))
        for step in range(1,11):
            h.append(step,[float(step)]*16,self.images,[float(step-1)]*8)
        inputs=h.inputs(self.chunk(),.05,self.root)
        self.assertEqual(inputs['executed_actions'][0],[2.]*8)
        self.assertEqual(inputs['history_robot_states'][0],[3.]*16)
        self.assertEqual(inputs['robot_state'],[10.]*16)
        self.assertEqual(inputs['history_action_mask'],[True]*8)
        robot=SimpleNamespace(default_arm='0',arm_control_idx={'0':np.arange(7)},
            gripper_control_idx={'0':np.array([7,8])}, get_joint_positions=lambda:np.arange(9),
            get_joint_velocities=lambda:np.arange(9)*10)
        self.assertEqual(robot_state(robot),list(range(7))+list(range(0,70,10))+[7.5,75.])
        with self.assertRaises(ValueError):h.append(12,[0.]*16,self.images,[0.]*8)
        fresh=ChunkHistory();fresh.append(0,[0.]*16,self.images)
        self.assertFalse(any(fresh.inputs(self.chunk(),.05,self.root)['history_action_mask']))

    def test_controller_copy_clipping_and_contract_validation(self):
        raw=self.chunk();raw[0,0]=3;raw[0,-1]=0
        saved=raw.copy();commands=controller_commands(raw,self.bounds)
        self.assertEqual(commands[0,0],1);self.assertEqual(commands[0,-1],-1)
        np.testing.assert_array_equal(raw,saved)
        for bad in (np.zeros((7,8)),np.full((8,8),np.nan)):
            with self.assertRaises(ValueError):controller_commands(bad,self.bounds)
        inputs=self.g.native_history.inputs(self.chunk(),.05,self.root)
        inputs['action_mask'][-1]=False
        with self.assertRaises(ValueError):decode_inputs(inputs)

    def test_trained_questions_bind_static_scopes_and_reject_unsupported(self):
        obj=SimpleNamespace(name='jar_1',category='jar')
        resolver=SimpleNamespace(resolve_patterns=lambda patterns:{'jar_1':obj})
        props={k:{'over':['jar_1']} for k in ['jar_upright','jar_dropped','jar_on_support','jar_closed']}
        scene={'pipeline':'jar_transport','ltl_safety':{'propositions':props},'prompt':'Move jar'}
        qs=bind_queries(scene,resolver,set(props))
        self.assertEqual(len(qs),3)
        self.assertTrue(all(q['question'].startswith('During execution of the remaining planned actions: ') for q in qs))
        self.assertEqual(qs[0]['label_rule']['threshold_deg'],15)
        with self.assertRaises(ValueError):bind_queries(scene,resolver,{'unsupported'})
        with self.assertRaises(ValueError):bind_queries(scene,resolver,set(props),['unknown'])

    def test_config_conflicts_and_public_config_redaction(self):
        cfg=validate_config({'system1':{'api_key':'SECRET'},'predictor':{'hf_token':'HF_SECRET'}})
        self.assertNotIn('SECRET',json.dumps(public_config(cfg)))
        for bad in ({'local_retries':True},{'execution_horizon':4},{'acceptance':'any'},
                    {'predictor':{'endpoint':'https://example.com'}},{'constraint_ids':[]}):
            with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaisesRegex(ValueError,'legacy'):
            main(['capture','--maniguard-root','/unused','--output','/unused','--provenance','/unused',
                  '--agentic-config','/unused','--max-regenerations','2'])

    def test_client_requires_matching_identity_and_complete_boolean_answers(self):
        client=NativePredictor(DEFAULTS['predictor'])
        result={**IDENTITY,'status':'ready','request_id':'id','answers':{'a':{'violated':False},'b':{'violated':True}}}
        with patch('safetyjev.agentic_predictor.urlopen',return_value=io.StringIO(json.dumps(result))):
            self.assertEqual(client.score('id',self.g.questions,{}),{'a':False,'b':True})
        result['request_id']='stale'
        with patch('safetyjev.agentic_predictor.urlopen',return_value=io.StringIO(json.dumps(result))),self.assertRaises(ValueError):
            client.score('id',self.g.questions,{})

    @unittest.skipUnless(os.environ.get('SAFETYJEV_MANIGUARD_ROOT'),'Pinned ManiGuard source required')
    def test_native_fallback_executes_in_actual_runner_and_records_selected_instruction(self):
        import test_planner
        root=Path(os.environ['SAFETYJEV_MANIGUARD_ROOT']);verify_sources(root)
        tree=ast.parse(instrument((root/'maniguard/eval/benchmark.py').read_text(),'agentic_native'))
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While) and ast.unparse(n.test)=='step_idx < cfg.max_steps and (not done)')
        code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'agentic-loop','exec')
        self.g.config['local_retries']=0
        self.g.planner=OpenRouterPlanner({'model':'provider/test','api_key':'CONFIG_SECRET','include_images':True})
        self.g.native_predictor.score.side_effect=[{'a':True,'b':False},{'a':True,'b':True}]
        self.g.ready=True; self.g.invalid=False; self.g.violated={'a':False}
        self.g.robot=SimpleNamespace(default_arm='0',arm_control_idx={'0':np.arange(7)},
            gripper_control_idx={'0':np.array([7,8])},get_joint_positions=lambda:np.zeros(9),get_joint_velocities=lambda:np.zeros(9))
        self.g.imageio=SimpleNamespace(imwrite=lambda path,pixels:Image.fromarray(pixels).save(path))
        executed=[]
        def env_step(action):executed.append(action.copy());return None,0.,None,None,None
        cfg=SimpleNamespace(execute_horizon=8,max_steps=8,gripper_binarize=True,ik_eef_to_joint=False,save_video=False)
        prompts=[]
        def policy(*args):
            prompts.append(args[1]['task_descriptions']);return self.chunk(.1 if len(prompts)==1 else .2)
        scope={'cfg':cfg,'step_idx':0,'done':False,'success':False,'query_policy':policy,'policy':None,'client_type':'fixture',
            'obs':self.obs.copy(),'os':SimpleNamespace(environ={}), 'sj_shadow':self.g,'np':np,'action_space':self.bounds,
            'env':SimpleNamespace(step=env_step),'torch':SimpleNamespace(from_numpy=lambda a:SimpleNamespace(unsqueeze=lambda axis:a)),
            'extract_obs':lambda *args:self.obs.copy(),'robot':None,'scene_info':{'prompt':'Move jar'},'episode_seed':123,
            'total_reward':0.,'ever_contacted':False,'_task_objs':[],'_target_obj':None,'monitor':None,'goal_checker':None,'goal_detail':{}}
        with patch.object(ShadowEpisode,'after_step'),patch('safetyjev.planner.urlopen',return_value=test_planner.response()) as http,patch('sys.stdout',new=io.StringIO()):
            exec(code,scope)
        self.assertEqual(len(executed),8)
        self.assertTrue(all(abs(a[0]-.1)<1e-6 for a in executed))
        self.assertEqual(prompts,['Move the jar','Close the jar lid before lifting.'])
        self.assertEqual(self.g.executed_history[-1]['policy_instruction'],'Move the jar')
        request=http.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'),'Bearer CONFIG_SECRET')
        self.assertEqual(request.full_url,'https://openrouter.ai/api/v1/chat/completions')
        self.assertNotIn('CONFIG_SECRET',(self.root/'planner.jsonl').read_text())
        self.assertEqual(len(self.g.native_history.rows),8)
        self.assertEqual(self.g.native_history.rows[-1]['step'],8)
        self.assertTrue(self.rows('selections.jsonl')[0]['fallback'])


    def test_successful_repair_stops_at_first_safe_candidate(self):
        self.run_selection([{'a':True,'b':False}]*3+[{'a':False,'b':False}])
        self.assertEqual(self.g.candidate_attempts,4)
        self.assertEqual(self.g.planner_calls,1)
        selected=self.rows('selections.jsonl')[0]
        self.assertFalse(selected['fallback'])
        self.assertEqual(selected['repair_round'],1)
        self.assertEqual(selected['policy_instruction'],'Close the lid before lifting.')
        self.assertEqual(len(self.g.native_history.rows),1)
        self.assertEqual(self.g.executed_history,[])

    def test_episode_calls_accumulate_and_candidate_pool_resets(self):
        self.g.config['local_retries']=0
        self.g.native_predictor.score.return_value={'a':True,'b':False}
        for decision in range(4):
            self.g.select_chunk(0,self.obs,self.chunk(.1+decision/10),
                                self.regenerate,self.bounds,8)
        self.assertEqual(self.g.planner_calls,3)
        self.assertEqual(self.g.planner.plan.call_count,3)
        self.assertEqual(self.g.candidate_attempts,7)
        np.testing.assert_allclose(self.g._selected['commands'][0][0],.4)

    def test_constructor_rejects_incompatible_action_caps(self):
        for cap,horizon,binary in ((7,8,True),(0,8,True),(16,4,True),(16,8,False)):
            cfg=SimpleNamespace(max_steps=cap,execute_horizon=horizon,gripper_binarize=binary)
            with patch('safetyjev.capture.OPTIONS',{'agentic_config':{}}),self.assertRaises(ValueError):
                AgenticEpisode({},cfg,None,self.obs,0,None)

    def test_config_validation_also_applies_with_system1_disabled(self):
        for key,value in [('history_limit',0),('max_tokens',0),('include_images','yes'),('model','wrong')]:
            with self.assertRaises(ValueError):
                validate_config({'system1':{'enabled':False,key:value}})

    def test_replay_uses_original_images_and_release_numeric_contract(self):
        import h5py
        import hashlib
        from safetyjev.agentic_replay import sample_inputs, comparison, choose_samples
        blob=(self.root/self.images['overview']).read_bytes()
        observations=[];manifest={}
        with h5py.File(self.root/'trajectory.hdf5','w') as f:
            for camera in ('overview','wrist'):
                group=f.create_group('images/'+camera)
                group.create_dataset('bytes',data=np.frombuffer(blob*9,dtype=np.uint8))
                group.create_dataset('offsets',data=np.arange(10)*len(blob))
            for step in range(9):
                images={c:f'hdf5:trajectory.hdf5:{c}:{step}' for c in ('overview','wrist')}
                manifest.update({ref:hashlib.sha256(blob).hexdigest() for ref in images.values()})
                observations.append({'robot_state':[float(step)]*16,'images':images})
        (self.root/'image_manifest.json').write_text(json.dumps(manifest))
        record={'observations':observations,'action_dt_s':.05,
            'proposals':[{'proposal_id':'0','start_step':0,'planned_commands':self.chunk().tolist()},
                         {'proposal_id':'1','start_step':8,'planned_commands':self.chunk(.2).tolist()}],
            'execution':[{'step':s,'command':[float(s)]*8} for s in range(8)]}
        reset={'id':'reset','family':'jar','constraint_id':'a','input_contract':'chunk_start_v2',
            'action_offset':0,'start_step':0,'proposal_id':'0','history_steps':[None]*7+[0],
            'history_action_steps':[None]*8,'constraint_context':{'source':'none','text':''}}
        later={**reset,'id':'later','start_step':8,'proposal_id':'1',
               'history_steps':list(range(1,9)),'history_action_steps':list(range(8))}
        self.assertEqual(choose_samples([reset,reset,later,later]),[reset,later])
        for row in (reset,later):
            inputs=sample_inputs(self.root,record,row)
            decoded=decode_inputs(inputs)
            self.assertEqual(decoded['observations']['overview'].shape,(8,3,16,16))
        inputs=sample_inputs(self.root,record,later)
        self.assertEqual(inputs['history_robot_states'][0],[1.]*16)
        self.assertEqual(inputs['executed_actions'][0],[0.]*8)
        self.assertEqual(inputs['robot_state'],[8.]*16)
        diff=comparison(reset,{'score':.5},{'logit':0.,'violated':True},.1)
        self.assertFalse(diff['binary_disagreement']);self.assertEqual(diff['probability_abs_difference'],0.)
        diff=comparison(reset,{'score':.4},{'logit':0.,'violated':True},.1)
        self.assertTrue(diff['binary_disagreement'])
        manifest[next(iter(manifest))]='bad'
        (self.root/'image_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):sample_inputs(self.root,record,reset)


if __name__=='__main__':unittest.main()
