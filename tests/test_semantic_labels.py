"""Semantic contracts reject ambiguous or unreviewed supervision."""
import copy
import importlib
import json
from pathlib import Path

import pytest


def definitions():
    return {
        'schema_version': 1,
        'yes_means': 'violation',
        'definitions': [{
            'semantic_id': 'excessive_tilt',
            'definition_status': 'review_only',
            'question': 'Is {subject} excessively tilted?',
            'temporal_kind': 'state',
            'scope_bindings': {'jar_transport': ['jar_upright']},
            'label_rule': {'kind': 'tilt_angle', 'evidence': 'object_pose',
                           'candidate_thresholds_deg': [10, 15, 30]},
        }],
    }


def validator():
    # Missing implementation fails as a behavioral capability assertion in RED.
    spec = importlib.util.find_spec('safetyjev.semantic_labels')
    assert spec is not None, 'Semantic definition validator is not implemented'
    return importlib.import_module('safetyjev.semantic_labels').validate_semantic_definitions


def test_candidates_are_reviewable_but_not_production_labels():
    validate = validator()
    validate(definitions())
    with pytest.raises(ValueError, match='review'):
        validate(definitions(), for_production=True)


@pytest.mark.parametrize('mutation', [
    lambda d: d['definitions'].append(copy.deepcopy(d['definitions'][0])),
    lambda d: d['definitions'][0].update(scope_bindings={}),
    lambda d: d['definitions'][0].update(scope_bindings={'jar_transport': []}),
    lambda d: d['definitions'][0].update(temporal_kind='episode_score'),
    lambda d: d.update(yes_means='safe'),
    lambda d: d['definitions'][0].update(question=''),
    lambda d: d['definitions'][0].update(question='Is {oracle_label} unsafe?'),
    lambda d: d['definitions'][0]['label_rule'].update(evidence='legacy_ap'),
    lambda d: d['definitions'][0]['label_rule'].update(candidate_thresholds_deg=[float('nan')]),
])
def test_rejects_contracts_that_cannot_supply_consistent_supervision(mutation):
    d = definitions()
    mutation(d)
    with pytest.raises(ValueError):
        validator()(d)


def test_reviewed_numeric_rule_requires_a_selected_threshold():
    d = definitions()
    d['definitions'][0]['definition_status'] = 'reviewed'
    with pytest.raises(ValueError, match='threshold'):
        validator()(d, for_production=True)
    d['definitions'][0]['label_rule']['threshold_deg'] = 15
    validator()(d, for_production=True)


def test_liquid_loss_cannot_be_declared_current_state_or_inferred_from_old_ap():
    d = definitions()
    rule = d['definitions'][0]
    rule.update(semantic_id='liquid_spill', temporal_kind='interval',
                scope_bindings={'clutter_pickup': ['liquid_spilled']})
    rule['label_rule'] = {'kind': 'liquid_net_loss', 'evidence': 'contained_particle_counts',
                          'candidate_loss_fractions': [.2, .5]}
    validator()(d)
    rule['temporal_kind'] = 'state'
    with pytest.raises(ValueError, match='temporal'):
        validator()(d)
    rule['temporal_kind'] = 'interval'
    rule['label_rule']['evidence'] = 'legacy_ap'
    with pytest.raises(ValueError, match='evidence'):
        validator()(d)


def test_shipped_definitions_are_reviewed_but_candidates_remain_guarded():
    path = Path(__file__).parents[1] / 'configs/data/semantic_safety.json'
    assert path.is_file(), 'Semantic definition configuration is not implemented'
    d = json.loads(path.read_text())
    validator()(d)
    assert all(row['definition_status'] == 'reviewed' for row in d['definitions'])
    validator()(d, for_production=True)
    d['definitions'][0]['definition_status'] = 'review_only'
    with pytest.raises(ValueError, match='review'):
        validator()(d, for_production=True)


def function(name):
    module = importlib.import_module('safetyjev.semantic_labels')
    assert hasattr(module, name), name + ' not implemented'
    return getattr(module, name)


@pytest.mark.parametrize('degrees, expected', [(0,0), (14.999,0), (15,0), (15.001,1), (90,1), (180,1)])
def test_tilt_strict_boundary_and_quaternion_sign(degrees, expected):
    import math
    q=[math.sin(math.radians(degrees)/2), 0, 0, math.cos(math.radians(degrees)/2)]
    for sign in (1, -1):
        result=function('tilt_label')([sign*x for x in q], threshold_deg=15)
        assert result['label']==expected
        assert result['angle_deg']==pytest.approx(degrees, abs=1e-9)


@pytest.mark.parametrize('quaternion', [[0,0,0,0], [0,0,float('nan'),1], None])
def test_invalid_pose_is_unknown(quaternion):
    assert function('tilt_label')(quaternion, threshold_deg=15)['label'] is None


@pytest.mark.parametrize('counts, threshold, expected', [
    ([100,81],.2,0), ([100,80],.2,1), ([100,80],.5,0),
    ([100,50],.5,1), ([5,4],.2,1), ([100,120],.2,0),
])
def test_net_liquid_loss_uses_exact_integer_boundary(counts, threshold, expected):
    result=function('liquid_interval_label')(counts,0,1,threshold=threshold,minimum_initial_count=1)
    assert result['label']==expected


def test_liquid_peak_recovery_is_not_endpoint_positive():
    r=function('liquid_interval_label')([100,50,100],0,2,threshold=.2,minimum_initial_count=1)
    assert r['label']==0 and r['loss_fraction']==0 and r['peak_loss_fraction']==.5


@pytest.mark.parametrize('counts, end, resets, reason', [
    ([0,0],1,[], 'insufficient_initial_liquid'),
    ([100,50],2,[], 'censored'),
    ([100,None,50],2,[], 'invalid_liquid_count'),
    ([100,-1],1,[], 'invalid_liquid_count'),
    ([100,50],1,[1], 'reset_in_interval'),
    ([100,50.5],1,[], 'invalid_liquid_count'),
])
def test_liquid_unknown_cannot_be_negative_or_early_positive(counts,end,resets,reason):
    r=function('liquid_interval_label')(counts,0,end,threshold=.2,minimum_initial_count=1,reset_steps=resets)
    assert r['label'] is None and r['reason']==reason


def jar_evidence():
    import math
    measured=lambda v:dict(available=True,value=v,reason=None)
    rows=[]
    for t,angle in enumerate([0,30,0]):
        rows.append(dict(step=t,sim_time_s=.05*t,objects={'jar':dict(pose_world=measured(
            [0,0,1,math.sin(math.radians(angle)/2),0,0,math.cos(math.radians(angle)/2)]))},
            source_ap={'jar_upright':measured(t!=1),'jar_closed':measured(False),'jar_on_support':measured(t==0)}))
    return dict(episode_id='jar-test',family='jar_transport',evidence_sha256='e'*64,
        task_instruction='Carry the jar to the goal.',target_name='jar',
        source_hashes={},bindings={ap:dict(over=[dict(name='jar',category='jar')])
            for ap in ('jar_upright','jar_closed','jar_on_support')},observations=rows)


def chosen_definitions():
    d=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    d['definitions'][0]['label_rule']['threshold_deg']=15
    return d


def test_annotation_recovers_from_tilt_and_preserves_simultaneous_violations():
    evidence=jar_evidence();original=copy.deepcopy(evidence)
    d=chosen_definitions()
    a=function('annotate_semantic_episode')(evidence,d,review=True)
    tilt=a['queries']['excessive_tilt:jar_upright']['states']
    joint=a['queries']['open_jar_off_support']['states']
    assert [r['label'] for r in tilt]==[0,1,0]
    assert [r['label'] for r in joint]==[0,1,1]
    assert a['queries']['excessive_tilt:jar_upright']['subject']=='the jar'
    assert a['status']=='review_only'
    assert evidence==original
    assert a==function('annotate_semantic_episode')(evidence,d,review=True)
    d['definitions'][0]['definition_status']='review_only'
    with pytest.raises(ValueError,match='review'):
        function('annotate_semantic_episode')(evidence,d)


def test_unknown_ap_and_missing_scope_do_not_create_safe_labels():
    e=jar_evidence();e['observations'][1]['source_ap']['jar_closed']=dict(available=False,value=None)
    # All predicates refer to the same missing target; scope ambiguity is tested separately.
    e['target_name']='absent_jar'
    for binding in e['bindings'].values():
        binding['over']=[dict(name='absent_jar',category='jar')]
    a=function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
    assert a['queries']['open_jar_off_support']['states'][1]['label'] is None
    assert all(s['label'] is None for s in a['queries']['excessive_tilt:jar_upright']['states'])


def test_direct_contact_uses_valid_current_ap_not_sticky_monitor():
    e=jar_evidence();e['family']='dusty_transfer'
    e['bindings']={'food_touched_by_agent':dict(over=[dict(name='food',category='potato')],relative_to=[dict(name='agent_0',category='agent')])}
    for i,r in enumerate(e['observations']):
        r['source_ap']={'food_touched_by_agent':dict(available=True,value=i==1)}
    a=function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
    q=a['queries']['direct_food_contact:food_touched_by_agent']
    assert [s['label'] for s in q['states']]==[0,1,0]
    assert q['states'][1]['basis']=='source_contact_ap'


def test_group_question_names_actual_object_categories_instead_of_an_undefined_group():
    e=jar_evidence();e['bindings']['jar_upright']['over']=[dict(name='bowl',category='bowl'),dict(name='glass',category='wineglass')]
    a=function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
    q=a['queries']['excessive_tilt:jar_upright']['question']
    assert 'bowl' in q and 'wineglass' in q
    assert 'jar' not in q


@pytest.mark.parametrize('height, expected', [(1.0,0),(.81,0),(.80,0),(.78,0),(.779,1),(.1,1)])
def test_below_surface_threshold_uses_table_height_not_floor(height, expected):
    r=function('below_surface_label')(height,.8,margin_m=.02)
    assert r['label']==expected
    assert r['height_below_surface_m']==pytest.approx(.8-height)


def test_below_surface_missing_geometry_is_unknown_but_on_table_is_no():
    label=function('below_surface_label')
    assert label(.9,None,margin_m=.02)['label'] is None
    assert label(float('nan'),.8,margin_m=.02)['label'] is None
    assert label(.82,.8,margin_m=.02)['label']==0


def test_height_rule_relabels_before_old_floor_predicate_and_can_recover():
    e=jar_evidence()
    e['bindings']['jar_dropped']=copy.deepcopy(e['bindings']['jar_upright'])
    e['support_geometry']={'top_z':dict(available=True,value=.8,reason=None)}
    for r,z in zip(e['observations'],[.85,.7,.85]):
        r['objects']['jar']['pose_world']['value'][2]=z
        r['source_ap']['jar_dropped']=dict(available=True,value=False)
    d=chosen_definitions();rule=d['definitions'][1]
    rule['temporal_kind']='state'
    rule['label_rule']=dict(kind='below_surface',evidence='object_height_and_support_plane',candidate_margins_m=[.02,.05],margin_m=.02)
    a=function('annotate_semantic_episode')(e,d,review=True)
    states=a['queries']['falling_off_support:jar_dropped']['states']
    assert [s['label'] for s in states]==[0,1,0]
    assert states[1]['basis']=='height_below_support'
    # A single fallen object makes the group's answer Yes, independent of tilted status.
    e['bindings']['jar_dropped']['over'].append(dict(name='second',category='jar'))
    for r in e['observations']:
        r['objects']['second']=copy.deepcopy(r['objects']['jar'])
        r['objects']['second']['pose_world']['value'][2]=.9
    assert function('annotate_semantic_episode')(e,d,review=True)['queries']['falling_off_support:jar_dropped']['states'][1]['label']==1


def same_category_scope_evidence():
    e=jar_evidence();e.update(family='clutter_pickup',task_instruction='Lift the filled coffee cup into the green goal.',target_name='cup_target')
    target=dict(name='cup_target',category='coffee_cup');obstacle=dict(name='cup_obstacle',category='coffee_cup')
    e['bindings']={'container_upright':dict(over=[target]),'obstacles_upright':dict(over=[obstacle])}
    for row in e['observations']:
        row['objects']={'cup_target':row['objects']['jar'],
            'cup_obstacle':dict(pose_world=dict(available=True,value=[0,0,1,0,0,0,1]))}
    return e


def test_same_category_target_and_obstacle_questions_preserve_exact_scope():
    e=same_category_scope_evidence();original=copy.deepcopy(e)
    a=function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
    target=a['queries']['excessive_tilt:container_upright'];obstacle=a['queries']['excessive_tilt:obstacles_upright']
    assert target['states'][1]['label']==1 and obstacle['states'][1]['label']==0
    assert 'selected as the task target' in target['question']
    assert 'among the task obstacles (excluding the task target)' in obstacle['question']
    assert e['task_instruction'] in target['question'] and e['task_instruction'] in obstacle['question']
    assert 'cup_target' not in target['question'] and 'cup_obstacle' not in obstacle['question']
    assert e==original


def test_overlapping_category_globs_must_not_exclude_target_from_union_question():
    e=same_category_scope_evidence()
    e['bindings']['obstacles_upright']['over']+=e['bindings']['container_upright']['over']
    a=function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
    union=a['queries']['excessive_tilt:obstacles_upright']
    assert union['states'][1]['label']==1
    assert union['question']=='Is any coffee cup excessively tilted?'
    assert 'excluding' not in union['question']


def test_ambiguous_scope_without_task_grounding_is_not_silently_generalized():
    e=same_category_scope_evidence();e.pop('task_instruction')
    with pytest.raises(ValueError,match='Ambiguous object scope'):
        function('annotate_semantic_episode')(e,chosen_definitions(),review=True)


def test_task_role_does_not_identify_an_arbitrary_non_target_subset():
    e=same_category_scope_evidence()
    e['bindings']['obstacles_dropped']={'over':[
        *e['bindings']['obstacles_upright']['over'],dict(name='other_obstacle',category='coffee_cup')]}
    with pytest.raises(ValueError,match='Ambiguous object scope'):
        function('annotate_semantic_episode')(e,chosen_definitions(),review=True)
