import unittest
import numpy as np
from safetyjev.predictor_judge_schema import make_action_window, validate_context, validate_episode_record, STATE_FEATURES
from predictor_judge_fixture import episode_fixture

class PredictorJudgeSchemaTests(unittest.TestCase):
    def test_suffix_and_padding_never_mutate_commands(self):
        commands=np.arange(64,dtype=np.float32).reshape(8,8);saved=commands.copy()
        for offset in (0,3,7):
            w=make_action_window(commands,offset)
            self.assertEqual(w['valid_steps'],8-offset)
            np.testing.assert_array_equal(w['actions'][:8-offset],commands[offset:])
            np.testing.assert_array_equal(w['action_mask'],np.arange(8)<8-offset)
            np.testing.assert_allclose(w['relative_times_s'][:8-offset],np.arange(8-offset)*.05)
            self.assertFalse(np.any(w['actions'][8-offset:]))
        np.testing.assert_array_equal(commands,saved)
    def test_real_zero_command_is_valid(self):
        w=make_action_window(np.zeros((1,8)),0)
        self.assertTrue(w['action_mask'][0]);self.assertEqual(w['valid_steps'],1)
    def test_rejects_invalid_windows(self):
        for commands,offset,kwargs in [(np.zeros((0,8)),0,{}),(np.zeros((9,8)),0,{}),
                (np.zeros((8,7)),0,{}),(np.zeros((8,8)),-1,{}),(np.zeros((8,8)),8,{}),
                (np.zeros((8,8)),True,{}),(np.full((8,8),np.nan),0,{}),
                (np.zeros((8,8)),0,{'dt_s':0}),(np.zeros((8,8)),0,{'dt_s':float('nan')})]:
            with self.subTest(offset=offset,kwargs=kwargs):
                with self.assertRaises(ValueError):make_action_window(commands,offset,**kwargs)
    def test_history_rule_does_not_require_oracle_input(self):
        validate_context({'source':'none','text':''},requires_history=False)
        validate_context({'source':'none','text':''},requires_history=True)
        validate_context({'source':'observed_history','text':'A was touched.'},requires_history=True)
        for c in [{'source':'oracle','text':'A was touched.'},
                  {'source':'observed_history','text':''}]:
            with self.assertRaises(ValueError):validate_context(c,requires_history=True)
    def test_episode_contract_detects_misalignment(self):
        e=episode_fixture();validate_episode_record(e);self.assertEqual(len(STATE_FEATURES),16)
        e['observations'][2]['step']=3
        with self.assertRaises(ValueError):validate_episode_record(e)
    def test_unknown_proposal_and_wrong_command_rejected(self):
        e=episode_fixture();e['execution'][0]['proposal_id']='missing'
        with self.assertRaises(ValueError):validate_episode_record(e)
        e=episode_fixture();e['execution'][0]['command'][0]=999
        with self.assertRaises(ValueError):validate_episode_record(e)
    def test_invalid_terminal_observation_is_preserved_without_poisoning_prefix(self):
        e=episode_fixture();e['observations'][-1]['valid']=False;e['observations'][-1]['robot_state']=[None]*16;e['oracle'][-1]['valid']=False
        validate_episode_record(e)
    def test_discarded_nonfinite_proposal_tail_is_not_a_model_input(self):
        e=episode_fixture();e['proposals'][0]['raw_actions'][-1]=[None]*8
        validate_episode_record(e)
