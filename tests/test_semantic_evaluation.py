"""Semantic state metrics do not borrow old monitor rejection lead times."""
from safetyjev.predictor_judge_eval import summarize_predictions


def test_semantic_confusion_counts_and_starting_status_are_separate_from_old_events():
    rows=[]
    for i,(label,score,started) in enumerate([(1,.9,False),(0,.9,False),(1,.1,True),(0,.1,True)]):
        rows.append(dict(id=str(i),label=label,score=score,nll=1.,constraint_id='tilt:jar',valid_steps=8,
            metadata=dict(semantic_id='excessive_tilt',starts_violated=started,episode_id='one',
                          family='jar_transport',first_violation_step=20,start_step=i)))
    r=summarize_predictions(rows)
    assert [r['micro'][k] for k in ('tp','fp','fn','tn')]==[1,1,1,1]
    assert r['by_semantic']['excessive_tilt']['n']==4
    assert r['by_starting_status']['not_violated']['tp']==1
    assert r['by_starting_status']['already_violated']['fn']==1
    assert r['events']['eligible']==0 and r['events']['recall'] is None


def test_classifier_collation_keeps_semantic_metadata_outside_model_inputs():
    import numpy as np
    from safetyjev.visual_dataset import collate_torch
    metadata={'family':'jar_transport','semantic_id':'excessive_tilt'}
    item={'inputs':{'question':'Is the jar excessively tilted?',
                   'observations':{'overview':np.zeros((1,8,8,3),dtype=np.uint8)}},
          'target':np.array([0.,1.]),'sample_id':'one','query_id':'tilt:jar','metadata':metadata}
    batch=collate_torch([item])
    assert batch.get('metadata')==[metadata]
    assert set(batch['inputs'])=={'questions','observations'}


def test_classifier_semantic_metrics_and_prediction_metadata(tmp_path):
    import json
    import torch
    from torch.utils.data import DataLoader
    from safetyjev.visual_train import evaluate_visual
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(1,1)
        def forward(self,features):
            assert torch.is_inference_mode_enabled()
            return torch.stack([features*0,features],-1)
    rows=[dict(inputs={'features':score},target=[1-y,y],sample_id=str(i),query_id='tilt:jar',
               metadata={'family':fam,'semantic_id':'excessive_tilt'})
          for i,(y,score,fam) in enumerate([(1,2.,'jar_transport'),(0,2.,'jar_transport'),(1,-2.,'stack_retrieve'),(0,-2.,'stack_retrieve')])]
    def collate(items):
        return {'inputs':{'features':torch.tensor([i['inputs']['features'] for i in items])},
                'targets':torch.tensor([i['target'] for i in items],dtype=torch.float32),'sample_ids':[i['sample_id'] for i in items],
                'query_ids':[i['query_id'] for i in items],'metadata':[i['metadata'] for i in items]}
    output=tmp_path/'predictions.jsonl'
    r,_,_=evaluate_visual(Model(),DataLoader(rows,batch_size=2,collate_fn=collate),output=output)
    assert r['by_semantic']['excessive_tilt']['n']==4
    assert [r['by_semantic']['excessive_tilt'][k] for k in ('tp','fp','fn','tn')]==[1,1,1,1]
    assert sum(x['n'] for x in r['by_family'].values())==4
    assert r['by_family']['jar_transport']['fp']==1
    assert [json.loads(s)['metadata'] for s in output.read_text().splitlines()]==[r['metadata'] for r in rows]
