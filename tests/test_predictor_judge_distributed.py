import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import test_predictor_judge_cache as cache_tests

class Model(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__();self.head=torch.nn.Linear(2,1);self.model_config={'fixture':True}
        self.backbone=torch.nn.Module();self.backbone.visual=torch.nn.Linear(1,1);self.backbone.visual.requires_grad_(False)
    def set_normalization(self,**kwargs):pass
    def forward(self,robot_state,**kwargs):
        score=self.head(robot_state[:,:2]).squeeze(-1);return torch.stack([score*0,score],-1)
    def save(self,path):
        Path(path).mkdir(parents=True);torch.save(self.state_dict(),Path(path)/'weights.pt')
    @classmethod
    def load(cls,path,**kwargs):
        m=cls();m.load_state_dict(torch.load(Path(path)/'weights.pt',weights_only=True));return m

def worker(rank,folder):
    from datetime import timedelta
    from safetyjev.predictor_judge_train import run
    root=Path(folder);os.environ.update(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),rank=rank,world_size=2,timeout=timedelta(seconds=30))
    config={'seed':42,'model':{'dtype':'float32'},'data':{'package':str(root/'package'),'batch_size':1,'num_workers':0,'frame_cache':str(root/'cache'),'sampling':{'samples_per_epoch':16}},
            'training':{'max_steps':2,'global_batch_size':4,'lr':.01,'head_lr':.01,'weight_decay':0.,'warmup_steps':0,'clip_grad_norm':1.,'brier_weight':.1,'eval_every':1,'save_every':1},
            'evaluation':{'max_batches':None,'validation_samples':3}}
    try:
        with patch('jev.predictor_judge_model.PredictorJudgeModel.from_pretrained',side_effect=lambda **kwargs:Model()),patch('jev.predictor_judge_model.PredictorJudgeModel.load',side_effect=Model.load):
            first=run(config,root/'run',device='cpu',stop_after=1)
            result=run(config,root/'run',device='cpu',resume=first['checkpoint'])
        assert result['training']['status']=='completed'
        assert result['sampling']['draws']==8
        assert result['test']['samples']==6
        assert result['validation']['samples']==6
        assert len((root/'run/final/test.jsonl').read_text().splitlines())==6
    finally:dist.destroy_process_group()

class DistributedJudgeTests(unittest.TestCase):
    def test_full_entrypoint_cache_sampler_resume_and_uneven_eval(self):
        from safetyjev.predictor_judge_cache import build_cache
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=cache_tests.JudgeCacheTests().package(root);build_cache(package,root/'cache')
            mp.spawn(worker,args=(temp,),nprocs=2,join=True)
            report=json.loads((root/'run/final/report.json').read_text())
            self.assertEqual(report['identity']['world_size'],2)
            self.assertEqual(report['sampling']['draws'],8)
