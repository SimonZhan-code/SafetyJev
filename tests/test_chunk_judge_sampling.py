import copy
from test_semantic_samples import pair
from safetyjev.semantic_data import build_semantic_samples
from safetyjev.predictor_judge_sampling import JudgeBatchSampler


def test_semantic_sampling_uses_episode_status_units_without_fake_events():
    e,a,d=pair([0]*9+[1]*8)
    rows=build_semantic_samples(e,a,d,task='predictor_judge',review=True,input_contract='chunk_start_v2')
    for r in rows:r.update(split='train',family='jar_transport',episode_safety='unsafe',negative_stratum='ordinary')
    # Deliberately no legacy first_violation_step, as in actual semantic packages.
    sampler=JudgeBatchSampler(rows,10,seed=1,epoch=0,samples_per_epoch=100,positive_fraction=.6)
    report=sampler.report()
    assert report['labels']=={'positive':60,'negative':40}
    assert report['distinct_positive_events']==0
    assert report['distinct_positive_sampling_units']==1
    assert report['available']['positive_events']==0
    assert list(sampler)==list(JudgeBatchSampler(rows,10,seed=1,epoch=0,samples_per_epoch=100,positive_fraction=.6))
