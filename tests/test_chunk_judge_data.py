"""Chunk-start temporal contract; fixtures do not establish real rollout coverage."""
import copy
import numpy as np
import pytest
from test_semantic_samples import pair
from safetyjev.semantic_data import build_semantic_samples


def chunk_rows(e,a,d):
    return build_semantic_samples(e,a,d,task='predictor_judge',review=True,
                                  input_contract='chunk_start_v2')


def test_only_real_chunk_starts_and_post_action_history():
    e,a,d=pair([0]*9+[1]+[0]*7)
    before=copy.deepcopy(e)
    rows=chunk_rows(e,a,d)
    assert [r['start_step'] for r in rows]==[0,8]
    assert [r['target'] for r in rows]==[[1.,0.],[0.,1.]]
    assert rows[0]['history_steps']==[None]*7+[0]
    assert rows[0]['history_action_steps']==[None]*8
    assert rows[1]['history_steps']==list(range(1,9))
    assert rows[1]['history_action_steps']==list(range(8))
    assert all(r['valid_steps']==8 and r['action_offset']==0 for r in rows)
    assert e==before


def test_irregular_replacement_is_a_boundary_and_missing_tail_is_not_negative():
    e,a,d=pair([0]*17)
    p=copy.deepcopy(e['proposals'][0]);p.update(proposal_id='replace',start_step=4)
    e['proposals'].append(p)
    for t in range(4,8):e['execution'][t].update(proposal_id='replace',offset=t-4,command=p['planned_commands'][t-4])
    rows=chunk_rows(e,a,d)
    assert [r['start_step'] for r in rows]==[0,4,8]
    assert rows[0]['target'] is None and rows[0]['label_reason']=='action_replaced'
    assert rows[1]['history_action_steps']==[None]*4+list(range(4))
    assert rows[2]['history_action_steps']==[None]*4+list(range(4,8))
    assert rows[2]['history_steps']==[None]*4+list(range(5,9))
    e,a,d=pair([0]*6)
    assert chunk_rows(e,a,d)[0]['target'] is None
    a['queries']['excessive_tilt:jar_upright']['states'][2]['label']=1
    assert chunk_rows(e,a,d)[0]['target']==[0.,1.]


def test_reset_clears_history_and_short_plan_is_explicitly_excluded():
    e,a,d=pair([0]*17);a['reset_steps']=[6]
    r=chunk_rows(e,a,d)[1]
    assert r['history_action_steps']==[None]*6+[6,7]
    assert r['history_steps']==[None]*6+[7,8]
    e,a,d=pair([0]*5)
    e['proposals'][0]['planned_commands']=e['proposals'][0]['planned_commands'][:4]
    e['proposals'][0]['planned_steps']=4
    r=chunk_rows(e,a,d)[0]
    assert r['target'] is None and r['label_reason']=='chunk_length_not_eight'


def test_runtime_input_assembler_uses_executed_actions_not_old_proposal_tail():
    from safetyjev.chunk_judge import chunk_numeric_inputs
    e,a,d=pair([0]*17)
    r=chunk_rows(e,a,d)[1]
    x=chunk_numeric_inputs(e,r)
    np.testing.assert_equal(x['executed_actions'],[v['command'] for v in e['execution'][:8]])
    assert x['history_action_mask'].tolist()==[True]*8
    assert x['robot_state'].tolist()==[8.]*16
    cold=chunk_numeric_inputs(e,chunk_rows(e,a,d)[0])
    assert not cold['history_action_mask'].any() and not cold['executed_actions'].any()


def test_chunk_package_direct_cache_and_batch(tmp_path):
    import json
    from test_semantic_package import definitions
    from test_source_episodes import source_fixture
    from safetyjev.predictor_judge_dataset import build_package,PredictorJudgeWindowDataset,collate_predictor_judge
    from safetyjev.predictor_judge_cache import build_cache
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,observation_count=17)
    package=tmp_path/'package'
    build_package([raw],package,group_splits={'jar_transport/task_0000':'train'},
        semantic_definitions=definitions(),input_contract='chunk_start_v2',train_episodes='all')
    meta=json.loads((package/'dataset_metadata.json').read_text())
    assert meta['input_contract']=='chunk_start_v2' and meta['history_frames']==8
    counts=json.loads((package/'composition.json').read_text())['query_windows']
    assert len(counts)==1 and counts[0]['positive']==1
    assert counts[0]['by_starting_status']['not_violated']['positive']==1
    direct=PredictorJudgeWindowDataset(package,'train')
    try:
        assert len(direct)==2
        item=direct[0];x=item['inputs']
        assert x['observations']['overview'].shape==(8,3,16,16)
        assert not x['history_action_mask'].any()
        assert x['history_mask'].tolist()==[False]*7+[True]
        assert set(x)=={'questions','observations','robot_state','executed_actions','history_robot_states','history_action_mask',
                        'remaining_actions','action_mask','action_dt_s','history_mask','constraint_context'}
        build_cache(package,tmp_path/'cache')
        cached=PredictorJudgeWindowDataset(package,'train',frame_cache=tmp_path/'cache')
        try:
            for key in ('executed_actions','history_robot_states','history_action_mask','robot_state'):
                np.testing.assert_equal(x[key],cached[0]['inputs'][key])
            batch=collate_predictor_judge([item,cached[0]])
            assert batch['inputs']['executed_actions'].shape==(2,8,8)
            for index in range(len(direct)):
                inputs=direct[index]['inputs'];other=cached[index]['inputs']
                for key in ('history_robot_states','executed_actions','robot_state','history_action_mask'):
                    np.testing.assert_array_equal(inputs[key],other[key])
                for camera in inputs['observations']:
                    np.testing.assert_array_equal(inputs['observations'][camera],other['observations'][camera])
            full=direct[1]['inputs']
            assert full['history_action_mask'].all()
            np.testing.assert_array_equal(full['history_robot_states'][-1],full['robot_state'])
            np.testing.assert_allclose(full['history_robot_states'][:,0],np.arange(1,9)/10)
            assert collate_predictor_judge([direct[0],direct[1]])['inputs']['history_robot_states'].shape==(2,8,16)
        finally:cached.close()
    finally:direct.close()
    meta['input_contract']='chunk_start_v1'
    (package/'dataset_metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Unsupported input contract'):
        PredictorJudgeWindowDataset(package,'train')


def test_history_states_are_post_action_causal_and_masked():
    from safetyjev.chunk_judge import chunk_numeric_inputs
    e,a,d=pair([0]*17)
    for t,o in enumerate(e['observations']):o['robot_state']=[float(t+j/100) for j in range(16)]
    row=chunk_rows(e,a,d)[1];x=chunk_numeric_inputs(e,row)
    np.testing.assert_equal(x['history_robot_states'],np.asarray([o['robot_state'] for o in e['observations'][1:9]],dtype=np.float32))
    np.testing.assert_array_equal(x['history_robot_states'][-1],x['robot_state'])
    assert x['history_robot_states'].dtype==np.float32
    for o in e['observations'][9:]:o['robot_state']=[999.]*16
    np.testing.assert_array_equal(x['history_robot_states'],chunk_numeric_inputs(e,row)['history_robot_states'])
    a['reset_steps']=[6];partial=chunk_numeric_inputs(e,chunk_rows(e,a,d)[1])
    assert not partial['history_robot_states'][:6].any()
    np.testing.assert_array_equal(partial['history_robot_states'][6:],x['history_robot_states'][6:])
    cold=chunk_numeric_inputs(e,chunk_rows(e,a,d)[0])
    assert not cold['history_robot_states'].any()
    np.testing.assert_array_equal(cold['robot_state'],np.asarray(e['observations'][0]['robot_state'],dtype=np.float32))
    e['observations'][7]['robot_state'][0]=float('nan')
    with pytest.raises(ValueError,match='state'):chunk_numeric_inputs(e,row)
    with pytest.raises(ValueError):chunk_numeric_inputs(e,{**row,'input_contract':'chunk_start_v1'})
