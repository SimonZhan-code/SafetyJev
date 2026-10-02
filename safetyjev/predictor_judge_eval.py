"""Offline future-violation metrics on observed, horizon-aligned labels."""
from collections import defaultdict
import json,math,time
from pathlib import Path
import torch
from .metrics import binary_metrics


@torch.inference_mode()
def evaluate_predictor_judge(model,loader,threshold=.5,*,output=None,max_batches=None):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    model.eval();pairs=[];by_query=defaultdict(list);by_length=defaultdict(list);nll=0.;elapsed=0.;records=[]
    for index,batch in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        device=getattr(getattr(model,'head',None),'weight',torch.zeros(1)).device
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter();logits=model(**batch['inputs'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed+=time.perf_counter()-start
        if logits.shape!=batch['targets'].shape or logits.shape[-1]!=2 or not torch.isfinite(logits).all():raise ValueError('Invalid predictor judge logits')
        logp=logits.float().log_softmax(-1).cpu();prob=logp.exp()[:,1].tolist()
        for i,p in enumerate(prob):
            target=batch['targets'][i].tolist()
            if target not in ([1.,0.],[0.,1.]):raise ValueError('Invalid binary label')
            y=int(target[1]);pair=(y,p);pairs.append(pair);nll-=float(logp[i,y])
            by_query[batch['query_ids'][i]].append(pair);by_length[str(batch['valid_steps'][i])].append(pair)
            records.append({'id':batch['sample_ids'][i],'constraint_id':batch['query_ids'][i],
                            'valid_steps':batch['valid_steps'][i],'label':y,'score':p})
    if not pairs:raise ValueError('No eligible evaluation samples')
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    return {'micro':binary_metrics(pairs,threshold),'nll':nll/len(pairs),
            'by_constraint':{k:binary_metrics(v,threshold) for k,v in by_query.items()},
            'by_valid_steps':{k:binary_metrics(v,threshold) for k,v in by_length.items()},
            'model_batch_seconds':elapsed,'threshold':threshold,'samples':len(pairs),
            'time_scope':'processor+model; excludes data decoding/transport; correlated per-step windows are not independent trials'}


def main(argv=None):
    import argparse
    from torch.utils.data import DataLoader
    from jev.predictor_judge_model import PredictorJudgeModel
    from .predictor_judge_dataset import PredictorJudgeWindowDataset,collate_predictor_judge
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--package',required=True)
    parser.add_argument('--split',choices=['validation','test'],default='test');parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=1);parser.add_argument('--device',default='cuda:0');parser.add_argument('--threshold',type=float,default=.5)
    args=parser.parse_args(argv)
    data=PredictorJudgeWindowDataset(args.package,args.split);model=PredictorJudgeModel.load(args.checkpoint,device=args.device)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=evaluate_predictor_judge(model,DataLoader(data,batch_size=args.batch_size,collate_fn=collate_predictor_judge),
                              args.threshold,output=output.with_suffix('.jsonl'))
    report.update(data_counts=data.summary['counts'],label_reasons=data.summary['label_reasons'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
