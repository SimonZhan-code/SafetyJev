import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(2,1);self.model_config={'fixture':True}
    def forward(self,features):
        score=self.head(features).squeeze(-1);return torch.stack([score*0,score],-1)
    def save(self,path):
        Path(path).mkdir();torch.save(self.state_dict(),Path(path)/'weights.pt')

def batches(rank=None):
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[-1.,1.],[1.,-1.],[2.,2.],[0.,-1.]])
    y=torch.tensor([[1.,0.],[0.,1.]]*4)
    def loader(epoch,start):
        if rank is None:
            rows=[{'inputs':{'features':x},'targets':y}]
        else:
            ids=torch.tensor([rank*2,rank*2+1,rank*2+4,rank*2+5])
            rows=[{'inputs':{'features':x[ix]},'targets':y[ix]} for ix in ids.split(2)]
        yield from rows[start:]
    return loader

def distributed_worker(rank, rendezvous, folder):
    from jev.visual_training import fit_updates
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        cfg=dict(max_steps=3,accumulation=2,lr=.01,head_lr=.01,weight_decay=0.,warmup_steps=0,clip_grad_norm=100.,brier_weight=.1,save_every=1,eval_every=0)
        torch.manual_seed(7);model=DistributedDataParallel(TinyModel())
        fit_updates(model,batches(rank),cfg,Path(folder)/'ddp',identity={})
        full={k:v.clone() for k,v in model.module.state_dict().items()}
        torch.manual_seed(7);model=DistributedDataParallel(TinyModel())
        first=fit_updates(model,batches(rank),cfg,Path(folder)/'resume',identity={},stop_after=1)
        if rank==0:
            with (Path(folder)/'resume/training.jsonl').open('a') as stream:stream.write(json.dumps({'step':2})+'\n')
        dist.barrier()
        model=DistributedDataParallel(TinyModel())
        fit_updates(model,batches(rank),cfg,Path(folder)/'resume' ,identity={},resume=first['checkpoint'])
        for key,value in full.items():torch.testing.assert_close(value,model.module.state_dict()[key],rtol=0,atol=0)
        if rank==0:torch.save(full,Path(folder)/'distributed.pt')
        from safetyjev.visual_train import evaluate_visual
        class EvaluationModel(torch.nn.Module):
            def __init__(self):super().__init__();self.head=torch.nn.Linear(1,1)
            def forward(self, features):return torch.stack([features*0,features],-1)
        records=[{'inputs':{'features':float(i-2)},'targets':[int(i<2),int(i>=2)],'sample_ids':str(i),'query_ids':'q'} for i in range(5)]
        from torch.utils.data import DataLoader
        loader=DataLoader(records[rank::2],batch_size=2,collate_fn=lambda b:{
            'inputs':{'features':torch.tensor([x['inputs']['features'] for x in b])},
            'targets':torch.tensor([x['targets'] for x in b]),'sample_ids':[x['sample_ids'] for x in b],'query_ids':['q']*len(b)})
        report,_,_=evaluate_visual(EvaluationModel(),loader,output=Path(folder)/'predictions.jsonl')
        assert report['evaluated']==5
        assert report['confusion']['fp']==0 and report['confusion']['fn']==0
        assert len((Path(folder)/'predictions.jsonl').read_text().splitlines())==5
    finally:dist.destroy_process_group()

def failing_worker(rank, rendezvous, folder):
    from jev.visual_training import fit_updates
    from torch.nn.parallel import DistributedDataParallel
    from datetime import timedelta
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=15))
    class FailureModel(TinyModel):
        def forward(self, features):
            if rank==1:raise RuntimeError('intentional rank failure')
            return super().forward(features)
    cfg=dict(max_steps=1,accumulation=2,lr=.01,head_lr=.01,weight_decay=0.,warmup_steps=0,clip_grad_norm=100.,brier_weight=.1,save_every=1,eval_every=0)
    try:fit_updates(DistributedDataParallel(FailureModel()),batches(rank),cfg,Path(folder)/'failed',identity={})
    finally:dist.destroy_process_group()

class DistributedTests(unittest.TestCase):
    def test_rank_failure_cannot_publish_successful_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(mp.ProcessRaisedException):
                mp.spawn(failing_worker,args=(str(root/'rendezvous'),folder),nprocs=2,join=True)
            self.assertFalse((root/'failed/training_status.json').exists())
            self.assertFalse((root/'failed/checkpoints').exists())

    def test_sampler_shards_one_stream_and_resume_cursor(self):
        from safetyjev.visual_distributed import GlobalBatchSampler,evaluation_indices,resolve_accumulation
        ranks=[list(GlobalBatchSampler(8,2,seed=7,epoch=0,rank=r,world_size=2)) for r in range(2)]
        self.assertEqual(sorted(i for rank in ranks for batch in rank for i in batch),list(range(8)))
        self.assertEqual(list(GlobalBatchSampler(8,2,seed=7,epoch=0,rank=1,world_size=2,start_batch=1)),ranks[1][1:])
        self.assertEqual(list(evaluation_indices(5,0,2)),[0,2,4])
        self.assertEqual(list(evaluation_indices(5,1,2)),[1,3])
        self.assertEqual(resolve_accumulation(128,1,4),32)
        with self.assertRaises(ValueError):resolve_accumulation(127,1,4)

    def test_two_rank_update_and_resume_match_global_batch_reference(self):
        from jev.visual_training import fit_updates
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            mp.spawn(distributed_worker,args=(str(root/'rendezvous'),folder),nprocs=2,join=True)
            torch.manual_seed(7);model=TinyModel()
            cfg=dict(max_steps=3,accumulation=1,lr=.01,head_lr=.01,weight_decay=0.,warmup_steps=0,clip_grad_norm=100.,brier_weight=.1,save_every=1,eval_every=0)
            fit_updates(model,batches(),cfg,root/'single',identity={})
            actual=torch.load(root/'distributed.pt',weights_only=True)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,actual[k],rtol=1e-5,atol=1e-7)
            records=[json.loads(l) for l in (root/'ddp/training.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),3)

if __name__=='__main__':unittest.main()
