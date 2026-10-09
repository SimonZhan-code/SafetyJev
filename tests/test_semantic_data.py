"""Temporal supervision must answer the time interval that was actually queried."""
import copy
import importlib
import json

import pytest


def module():
    assert importlib.util.find_spec('safetyjev.semantic_data') is not None, 'Semantic interval implementation missing'
    return importlib.import_module('safetyjev.semantic_data')


def annotation(labels):
    return dict(schema_version=1,episode_id='example',status='review_only',definition_sha256='d'*64,
        evidence_sha256='e'*64,queries={'tilt:target':dict(semantic_id='excessive_tilt',
            temporal_kind='state',question='Is the jar excessively tilted?',
            states=[dict(step=t,label=y,reason='source_invalid' if y is None else 'evaluated')
                    for t,y in enumerate(labels)])})


def test_classifier_current_and_predictor_future_are_different_questions():
    evaluate=module().evaluate_semantic_interval;a=annotation([1,0,0])
    assert evaluate(a,'tilt:target',0,0,task='classifier')['label']==1
    r=evaluate(a,'tilt:target',0,2,task='predictor_judge')
    assert r['label']==0 and r['starts_violated'] is True
    a=annotation([0,1,0])
    assert evaluate(a,'tilt:target',0,0,task='classifier')['label']==0
    assert evaluate(a,'tilt:target',0,2,task='predictor_judge')['label']==1


def test_positive_observed_prefix_survives_termination_but_not_a_prior_gap():
    evaluate=module().evaluate_semantic_interval
    assert evaluate(annotation([0,1]),'tilt:target',0,8,task='predictor_judge')['label']==1
    r=evaluate(annotation([0,0]),'tilt:target',0,8,task='predictor_judge')
    assert r['label'] is None and r['reason']=='censored'
    r=evaluate(annotation([0,None,1]),'tilt:target',0,2,task='predictor_judge')
    assert r['label'] is None and r['reason']=='source_invalid'
    a=annotation([0,0,1]);del a['queries']['tilt:target']['states'][1]
    assert evaluate(a,'tilt:target',0,2,task='predictor_judge')['label'] is None


def liquid_annotation(counts):
    a=annotation([]);a['queries']={'spill:cup':dict(semantic_id='liquid_spill',temporal_kind='interval',
        objects=[dict(name='cup')],label_rule=dict(loss_fraction=.2,minimum_initial_count=1),
        states=[dict(step=t,label=None,contained_counts=dict(available=True,value={'cup':n}))
                for t,n in enumerate(counts)])}
    return a


def test_liquid_needs_complete_endpoints_and_classifier_history():
    evaluate=module().evaluate_semantic_interval
    a=liquid_annotation([100,50,100])
    assert evaluate(a,'spill:cup',0,2,task='predictor_judge')['label']==0
    assert evaluate(a,'spill:cup',0,1,task='predictor_judge')['label']==1
    r=evaluate(a,'spill:cup',0,3,task='predictor_judge')
    assert r['label'] is None and r['reason']=='censored'
    r=evaluate(a,'spill:cup',0,0,task='classifier')
    assert r['label'] is None and r['reason']=='input_history_required'


def test_query_scope_and_indices_must_be_unambiguous():
    evaluate=module().evaluate_semantic_interval;a=annotation([0,1])
    with pytest.raises(KeyError):evaluate(a,'excessive_tilt',0,1,task='predictor_judge')
    for start,end,task in [(0,0,'predictor_judge'),(-1,1,'classifier'),(0,1,'wrong')]:
        with pytest.raises(ValueError):evaluate(a,'tilt:target',start,end,task=task)


def test_saved_annotation_digest_detects_label_and_question_changes(tmp_path):
    mod=module();a=annotation([0,1]);path=tmp_path/'annotation.json'
    original=copy.deepcopy(a);mod.write_annotation(path,a)
    assert a==original
    loaded=mod.read_annotation(path);assert loaded['queries']==a['queries']
    loaded['queries']['tilt:target']['states'][0]['label']=1
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError,match='digest'):mod.read_annotation(path)
