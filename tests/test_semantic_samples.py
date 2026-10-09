"""Causal samples share semantic GT while preserving their distinct input horizons."""
import copy
import json
from pathlib import Path

import pytest
from predictor_judge_fixture import episode_fixture
from safetyjev import semantic_data
from safetyjev.semantic_labels import annotate_semantic_episode


def pair(labels, *, steps=None):
    e=episode_fixture(steps=len(labels)-1 if steps is None else steps,violation_step=1)
    d=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    d['definitions']=[x for x in d['definitions'] if x['semantic_id']=='excessive_tilt']
    d['definitions'][0]['label_rule']['threshold_deg']=15
    evidence=dict(episode_id=e['episode_id'],family='jar_transport',source_hashes={},evidence_sha256='fixture',
        bindings={'jar_upright':{'over':[dict(name='jar',category='jar')]}},observations=[
            dict(step=t,sim_time_s=t*.05,objects={'jar':{'pose_world':dict(available=True,value=[0,0,1,0,0,0,1])}})
            for t in range(len(labels))])
    a=annotate_semantic_episode(evidence,d,review=True)
    for s,label in zip(a['queries']['excessive_tilt:jar_upright']['states'],labels):
        s['label']=label;s['reason']='evaluated' if label is not None else 'orientation_unavailable'
    return e,a,d


def build(e,a,d,task,**kwargs):
    assert hasattr(semantic_data,'build_semantic_samples'), 'Semantic sample builder not implemented'
    return semantic_data.build_semantic_samples(e,a,d,task=task,review=True,**kwargs)


def test_current_and_future_labels_use_their_actual_distinct_time_ranges():
    e,a,d=pair([0,0,1]+[0]*14)
    original=copy.deepcopy((e,a,d))
    current=build(e,a,d,'classifier')
    future=build(e,a,d,'predictor_judge')
    assert len(current)==17 and len(future)==16
    assert [r['target'] for r in current[:4]]==[[1.,0.],[1.,0.],[0.,1.],[1.,0.]]
    assert future[0]['target']==[0.,1.]
    assert future[2]['target']==[1.,0.] and future[2]['starts_violated'] is True
    assert current[2]['history_steps']==[2] and current[2]['label_interval']==[2,2]
    assert future[2]['history_steps']==[0,1,2] and future[2]['label_interval']==[2,8]
    assert future[0]['history_steps']==[None,None,0]
    assert [r['valid_steps'] for r in future]==list(range(8,0,-1))*2
    assert (e,a,d)==original
    assert all('30 degrees' not in r['question'] for r in future)
    assert all(r['constraint_context']=={'source':'none','text':''} for r in future)


def test_future_new_chunk_is_not_spliced_into_current_horizon():
    e,a,d=pair([0]*9+[1]+[0]*7)
    rows=build(e,a,d,'predictor_judge')
    assert rows[7]['end_step']==8 and rows[7]['valid_steps']==1 and rows[7]['target']==[1.,0.]
    assert rows[8]['proposal_id']=='p8' and rows[8]['target']==[0.,1.]


def test_early_termination_retains_only_observed_state_positives():
    e,a,d=pair([0,0,1,0,0,0])
    rows=build(e,a,d,'predictor_judge')
    assert rows[0]['end_step']==8 and rows[0]['target']==[0.,1.]
    assert rows[2]['target'] is None and rows[2]['label_reason']=='censored'
    assert rows[2]['observed_end']==5


def test_action_replacement_cannot_use_new_proposals_outcome():
    e,a,d=pair([0]*6+[1]+[0]*10)
    p=copy.deepcopy(e['proposals'][0]);p.update(proposal_id='replace',start_step=4)
    e['proposals'].append(p)
    for t in range(4,8):e['execution'][t].update(proposal_id='replace',offset=t-4,command=p['planned_commands'][t-4])
    rows=build(e,a,d,'predictor_judge')
    assert rows[0]['target'] is None and rows[0]['label_reason']=='action_replaced'
    assert rows[0]['observed_end']==4
    assert rows[4]['target']==[0.,1.]


def test_reset_or_missing_annotation_cannot_bridge_to_later_positive():
    e,a,d=pair([0]*6+[1]+[0]*10)
    a['reset_steps']=[4]
    rows=build(e,a,d,'predictor_judge')
    assert rows[0]['target'] is None and rows[0]['label_reason']=='reset_in_interval'
    a.pop('reset_steps');a['queries']['excessive_tilt:jar_upright']['states'][4]['label']=None
    a['queries']['excessive_tilt:jar_upright']['states'][4]['reason']='orientation_unavailable'
    assert build(e,a,d,'predictor_judge')[0]['target'] is None


def test_changed_definition_or_episode_is_rejected_before_constructing_samples():
    e,a,d=pair([0]*17)
    bad=copy.deepcopy(d);bad['definitions'][0]['question']='Has the jar tilted?'
    with pytest.raises(ValueError,match='definition'):build(e,a,bad,'classifier')
    a['episode_id']='different'
    with pytest.raises(ValueError,match='episode'):build(e,a,d,'predictor_judge')


def test_current_classifier_excludes_interval_question_without_adding_history():
    e,a,d=pair([0]*17)
    q=copy.deepcopy(a['queries']['excessive_tilt:jar_upright'])
    q.update(temporal_kind='interval',semantic_id='liquid_spill')
    a['queries']={'spill':q}
    rows=build(e,a,d,'classifier')
    assert all(r['target'] is None and r['label_reason']=='input_history_required' for r in rows)
    assert all(len(r['history_steps'])==1 for r in rows)


def test_invalid_commands_are_excluded_and_oracle_values_not_copied_into_rows():
    e,a,d=pair([0]*12,steps=11)
    e['proposals'][1]['planned_commands'][3]=[None]*8
    e['proposals'][1]['raw_actions'][3]=[None]*8
    e['result']['status']='numerical_failed'
    rows=build(e,a,d,'predictor_judge')
    assert rows[8]['target'] is None and rows[8]['label_reason']=='invalid_actions'
    assert not any(key in rows[0] for key in ('source_ap','oracle','angles','heights','contained_counts'))
