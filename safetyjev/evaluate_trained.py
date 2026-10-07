"""Offline single-device/DDP evaluation for both trained SafetyJev model types."""
import argparse,json,os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from .visual_train import evaluate_visual,_rank_zero_call,verify_package
from .predictor_judge_eval import evaluate_predictor_judge


def run(args):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));device=args.device
    if world>1 and not dist.is_initialized():
        local=int(os.environ['LOCAL_RANK'])
        if device.startswith('cuda'):torch.cuda.set_device(local);device=f'cuda:{local}'
        dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',timeout=timedelta(minutes=10))
    if args.batch_size<1 or args.workers<0:raise ValueError('Invalid batch/workers')
    package=Path(args.package);output=Path(args.output)
    if args.task=='predictor_judge':
        from .predictor_judge_dataset import PredictorJudgeWindowDataset,PreparedPredictorJudgeCollator
        from jev.predictor_judge_model import PredictorJudgeModel
        data=PredictorJudgeWindowDataset(package,args.split,frame_cache=args.frame_cache)
        model=PredictorJudgeModel.load(args.checkpoint,device=device)
        for key in ['history_frames','max_actions','state_features']:
            if model.model_config.get(key)!=data.summary[key]:raise ValueError('Model/data contract differs: '+key)
        collator=PreparedPredictorJudgeCollator(model.model_config)
    else:
        from .visual_dataset import APWindowDataset,PreparedVisualCollator
        from jev.visual_model import VisualDecisionModel
        _rank_zero_call(lambda:verify_package(package,verify_raw=not bool(args.frame_cache)))
        data=APWindowDataset(package,args.split,frame_cache=args.frame_cache)
        model=VisualDecisionModel.load(args.checkpoint,device=device);collator=PreparedVisualCollator(model.model_config)
    options={'num_workers':args.workers,'collate_fn':collator,'pin_memory':device.startswith('cuda')}
    if args.workers:options.update(persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
    subset=Subset(data,list(range(rank,len(data),world)))
    loader=DataLoader(subset,batch_size=args.batch_size,shuffle=False,**options)
    if args.task=='predictor_judge':
        report=evaluate_predictor_judge(model,loader,threshold=args.threshold,output=output.with_suffix('.jsonl'))
        report.update(data_counts=data.summary['counts'],label_reasons=data.summary['label_reasons'])
    else:
        if args.threshold!=.5:raise ValueError('Classifier reports use the fixed 0.5 No/Yes threshold')
        report,_,_=evaluate_visual(model,loader,output=output.with_suffix('.jsonl'))
    report.update(task=args.task,split=args.split,expected_samples=len(data),checkpoint=str(args.checkpoint))
    from .model_export import checkpoint_hashes,sha256
    def provenance():
        return {'checkpoint_sha256':checkpoint_hashes(args.checkpoint),
                'package_sha256':sha256(package/'dataset_metadata.json'),
                'split_sha256':sha256(package/(args.split+'.jsonl'))}
    report.update(_rank_zero_call(provenance))
    count=report['samples'] if args.task=='predictor_judge' else report['evaluated']
    if count!=len(data):raise ValueError('Evaluation coverage differs from the complete split')
    def publish():
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    _rank_zero_call(publish);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['classifier','predictor_judge'],required=True)
    for key in ['checkpoint','package','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--frame-cache')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--device',default='cuda:0');p.add_argument('--threshold',type=float,default=.5)
    args=p.parse_args(argv)
    try:
        report=run(args)
        if int(os.environ.get('RANK','0'))==0:print(json.dumps(report,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
