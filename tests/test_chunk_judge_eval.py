import pytest


def rows():
    return [dict(id=str(i),constraint_id=q,valid_steps=8,label=y,score=p,nll=.2,
        metadata=dict(episode_id='e',start_step=t,end_step=t+8,proposal_id='p'+str(t),
                      input_contract='chunk_start_v2',semantic_id='tilt',starts_violated=t>0))
        for i,(t,q,y,p) in enumerate([(0,'a',1,.8),(0,'b',0,.1),(8,'a',0,.9),(8,'b',0,.2)])]


def test_pair_headline_and_boundary_aggregation_are_distinct():
    from safetyjev.chunk_judge_eval import summarize_chunk_predictions
    report=summarize_chunk_predictions(rows(),threshold=.5)
    assert report['pairs']['confusion']==dict(tp=1,fp=1,tn=2,fn=0)
    assert report['pairs']['accuracy']==.75
    assert report['chunks']['confusion']==dict(tp=1,fp=1,tn=0,fn=0)
    assert report['chunks']['accuracy']==.5
    assert report['currently_safe_pairs']['n']==2
    assert report['episodes']['detected_at_first_positive_chunk']==1


def test_unknown_query_cannot_create_negative_chunk_or_independent_event():
    from safetyjev.chunk_judge_eval import summarize_chunk_predictions
    records=rows();records[0].update(label=None,score=.1)
    report=summarize_chunk_predictions(records,threshold=.5)
    assert report['excluded_pairs']==1 and report['unknown_chunks']==1
    assert report['chunks']['n']==1
    assert 'events' not in report
    with pytest.raises(ValueError):summarize_chunk_predictions(records+[records[0]])


def test_shadow_calls_once_per_boundary_with_all_queries_and_no_label_inputs(tmp_path):
    import json,numpy as np,torch
    from types import SimpleNamespace
    from safetyjev.chunk_judge_eval import replay_episode_chunks
    from safetyjev.chunk_judge import chunk_numeric_inputs
    from safetyjev.predictor_judge_dataset import _evaluation_metadata
    from test_semantic_samples import pair
    from safetyjev.semantic_data import build_semantic_samples
    e,a,d=pair([0]*9+[1]+[0]*7)
    source=build_semantic_samples(e,a,d,task='predictor_judge',review=True,input_contract='chunk_start_v2')
    samples=[]
    for r in source:
        r.update(split='validation',family='jar_transport')
        samples.extend([r,{**r,'id':r['id']+'b','constraint_id':'other'}])
    # One query has unknown supervision but still valid causal inputs.
    samples[1]={**samples[1],'target':None,'label_reason':'censored'}
    path=tmp_path/'validation.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in samples if r['target'] is not None))
    (tmp_path/'excluded.jsonl').write_text(json.dumps(samples[1])+'\n')
    def sample(row):
        return dict(inputs=dict(**chunk_numeric_inputs(e,row),questions=row['question'],
            observations={c:np.zeros((8,3,8,8),dtype=np.uint8) for c in ('overview','wrist')},
            constraint_context=row['constraint_context']),target=row['target'],sample_id=row['id'],
            query_id=row['constraint_id'],valid_steps=8,metadata=_evaluation_metadata(row))
    data=SimpleNamespace(summary={'input_contract':'chunk_start_v2'},frame_cache=None,path=path,package=tmp_path,sample=sample)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(1,1);self.calls=[]
        def forward(self,**inputs):
            assert set(inputs)=={'questions','observations','robot_state','remaining_actions','action_mask','action_dt_s',
                                 'history_mask','constraint_context','executed_actions','history_robot_states','history_action_mask'}
            self.calls.append(len(inputs['questions']))
            return torch.zeros(len(inputs['questions']),2)
    model=Model()
    from unittest.mock import patch
    with patch('safetyjev.chunk_judge_eval.time.perf_counter',side_effect=[0.,1.,3.,4.,10.,11.,14.,16.]):
        report=replay_episode_chunks(model,data,output=tmp_path/'replay.json')
    assert model.calls==[2,2] and report['calls']==2
    assert report['excluded_pairs']==1 and report['latency_s']['steady_calls']==1

    assert report['latency_s']['total_model_call_seconds']==5.
    assert report['latency_s']['total_boundary_seconds']==10.
