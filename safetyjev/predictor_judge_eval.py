"""Offline future-violation metrics on observed, horizon-aligned labels."""
from collections import defaultdict
import json,math,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from .metrics import binary_metrics
from .visual_train import _rank_zero_call,answer_metrics


def summarize_predictions(records,threshold=.5):
    pairs=[];by_query=defaultdict(list);by_length=defaultdict(list);by_family=defaultdict(list)
    nll=0.;events={};safe_episodes={};seen=set()
    for row in records:
        if row['id'] in seen:raise ValueError('Duplicate evaluation sample')
        seen.add(row['id']);y=row['label'];p=row['score'];pair=(y,p)
        if y not in (0,1) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid prediction')
        pairs.append(pair);nll+=row['nll']
        by_query[row['constraint_id']].append(pair);by_length[str(row['valid_steps'])].append(pair)
        meta=row.get('metadata',{});family=meta.get('family')
        if family is not None:by_family[family+'/'+row['constraint_id']].append(pair)
        eid=meta.get('episode_id')
        if eid and meta.get('episode_safety')=='safe':safe_episodes[eid]=safe_episodes.get(eid,False) or p>=threshold
        if y and eid and meta.get('first_violation_step') is not None:
            key=(eid,row['constraint_id'],meta['first_violation_step'])
            lead=meta['first_violation_step']-meta['start_step']
            events[key]=max(events.get(key,0),lead if p>=threshold else 0)
    if not pairs:raise ValueError('No eligible evaluation samples')
    return {'micro':binary_metrics(pairs,threshold),'nll':nll/len(pairs),
            'answers':answer_metrics([[1-y,y] for y,p in pairs],[p for y,p in pairs],threshold=threshold),
            'by_constraint':{k:binary_metrics(v,threshold) for k,v in by_query.items()},
            'by_family_constraint':{k:binary_metrics(v,threshold) for k,v in by_family.items()},
            'by_valid_steps':{k:binary_metrics(v,threshold) for k,v in by_length.items()},
            'events':{'eligible':len(events),'detected':sum(v>0 for v in events.values()),
                      'recall':sum(v>0 for v in events.values())/len(events) if events else None,
                      'mean_first_detected_lead_steps':sum(v for v in events.values() if v>0)/sum(v>0 for v in events.values()) if any(events.values()) else None},
            'safe_source_episodes':{'n':len(safe_episodes),'any_false_alarm':sum(safe_episodes.values()),
                                    'any_false_alarm_rate':sum(safe_episodes.values())/len(safe_episodes) if safe_episodes else None},
            'threshold':threshold,'samples':len(pairs),
            'interpretation':'Correlated per-step windows; event statistics cover events with eligible positive windows only. Safe-source episode rates need a complete uncapped split.'}


@torch.inference_mode()
def evaluate_predictor_judge(model,loader,threshold=.5,*,output=None,max_batches=None):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    distributed=dist.is_initialized();rank=dist.get_rank() if distributed else 0;world=dist.get_world_size() if distributed else 1
    if distributed and output is None:raise ValueError('Distributed evaluation needs a shared output path')
    model.eval();elapsed=0.;records=[];path=Path(output) if output else None
    part=path.with_name(path.name+f'.rank{rank}') if distributed else path
    if part:part.parent.mkdir(parents=True,exist_ok=True)
    writer=part.open('w') if part else None
    try:
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
                row={'id':batch['sample_ids'][i],'constraint_id':batch['query_ids'][i],
                     'valid_steps':batch['valid_steps'][i],'label':int(target[1]),'score':p,'nll':-float(logp[i,int(target[1])]),
                     'metadata':batch.get('metadata',[{}]*len(prob))[i]}
                if writer:writer.write(json.dumps(row,allow_nan=False)+'\n')
                if not distributed:records.append(row)
    finally:
        if writer:writer.close()
    if not distributed:report=summarize_predictions(records,threshold)
    else:
        timings=[None]*world;dist.all_gather_object(timings,elapsed);elapsed=max(timings)
        def merge():
            temporary=path.with_suffix(path.suffix+'.tmp')
            def rows():
                with temporary.open('w') as out:
                    for r in range(world):
                        with path.with_name(path.name+f'.rank{r}').open() as stream:
                            for line in stream:out.write(line);yield json.loads(line)
            result=summarize_predictions(rows(),threshold);os.replace(temporary,path)
            for r in range(world):path.with_name(path.name+f'.rank{r}').unlink()
            return result
        report=_rank_zero_call(merge)
    report.update(model_batch_seconds=elapsed,time_scope='forward plus tensor transfer; worker preprocessing, loader and merging excluded')
    return report

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
