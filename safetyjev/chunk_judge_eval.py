"""Compact query/chunk metrics and ordered shadow replay (no intervention claim)."""
from collections import defaultdict
import json,math,time
from pathlib import Path
from .metrics import ratio,quantile


def confusion(pairs,threshold):
    counts={k:0 for k in ('tp','fp','tn','fn')}
    for y,p in pairs:counts[('t' if y==(p>=threshold) else 'f')+('p' if p>=threshold else 'n')]+=1
    tp,fp,tn,fn=(counts[k] for k in ('tp','fp','tn','fn'))
    return dict(n=len(pairs),accuracy=ratio(tp+tn,len(pairs)),recall=ratio(tp,tp+fn),
                precision=ratio(tp,tp+fp),confusion=counts)


def summarize_chunk_predictions(records,threshold=.5):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    seen=set();groups=defaultdict(list);pairs=[];current=[];ongoing=[];excluded=0
    for row in records:
        if row['id'] in seen:raise ValueError('Duplicate prediction')
        seen.add(row['id']);meta=row['metadata'];y=row['label'];p=row['score']
        if meta.get('input_contract')!='chunk_start_v2':raise ValueError('Chunk replay requires chunk input contract')
        if y not in (None,0,1) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid prediction')
        groups[(meta['episode_id'],meta['start_step'],meta['proposal_id'])].append((y,p))
        if y is None:excluded+=1
        else:
            pairs.append((y,p))
            if meta.get('starts_violated') is False:current.append((y,p))
            elif meta.get('starts_violated') is True:ongoing.append((y,p))
    chunks=[];unknown=0;episodes=defaultdict(list)
    for (eid,t,pid),values in sorted(groups.items()):
        labels=[y for y,p in values];score=max(p for y,p in values)
        target=1 if 1 in labels else (None if None in labels else 0)
        episodes[eid].append((t,target,score))
        if target is None:unknown+=1
        else:chunks.append((target,score))
    positive=detected=0
    for values in episodes.values():
        first=next((r for r in values if r[1]==1),None)
        if first is not None:
            positive+=1;detected+=first[2]>=threshold
    return dict(pairs=confusion(pairs,threshold),currently_safe_pairs=confusion(current,threshold),
        already_violated_pairs=confusion(ongoing,threshold),chunks=confusion(chunks,threshold),
        excluded_pairs=excluded,unknown_chunks=unknown,threshold=threshold,
        episodes=dict(n=len(episodes),with_positive_chunk=positive,detected_at_first_positive_chunk=detected,
                      first_positive_chunk_recall=ratio(detected,positive)),
        scope='Observed chunk/query predictions; shadow replay does not establish accident prevention or safe-episode false-alarm rate')


def replay_episode_chunks(model,dataset,*,threshold=.5,output=None):
    """Once per recorded boundary, batch all queries and continue regardless of alarm.

    Requires direct raw access because excluded-supervision rows also need a model
    call. Labels affect scoring only; excluded input-invalid rows are counted apart.
    """
    import torch
    from .predictor_judge_dataset import collate_predictor_judge
    if dataset.summary.get('input_contract')!='chunk_start_v2' or dataset.frame_cache:
        raise ValueError('Shadow replay requires uncached chunk package and original media')
    from .package_layout import split_file
    split=dataset.split;groups=defaultdict(list);invalid=0
    for path in (dataset.path,split_file(dataset.package,'excluded',dataset.summary)):
        with path.open('rb') as stream:
            for line in stream:
                row=json.loads(line)
                if row['split']!=split:continue
                if row['label_reason'] in ('chunk_length_not_eight','invalid_actions','invalid_observation','reset_in_history'):
                    invalid+=1;continue
                groups[(row['episode_id'],row['start_step'],row['proposal_id'])].append(row)
    records=[];timings=[];forwards=[];model.eval()
    device=model.head.weight.device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with torch.inference_mode():
        for key,rows in sorted(groups.items()):
            sync();started=time.perf_counter();items=[]
            for row in rows:
                item=dataset.sample(row)
                # Collator needs a target-shaped placeholder, never passed to model.
                if item['target'] is None:item={**item,'target':[1.,0.]}
                items.append(item)
            batch=collate_predictor_judge(items);sync();before=time.perf_counter()
            logits=model(**batch['inputs']);sync();after=time.perf_counter()
            if logits.shape!=(len(rows),2) or not torch.isfinite(logits).all():raise ValueError('Invalid model output')
            scores=logits.float().softmax(-1)[:,1].cpu().tolist()
            timings.append(time.perf_counter()-started);forwards.append(after-before)
            for row,p in zip(rows,scores):
                from .predictor_judge_dataset import _evaluation_metadata
                records.append(dict(id=row['id'],constraint_id=row['constraint_id'],valid_steps=8,
                    label=None if row['target'] is None else int(row['target'][1]),score=p,metadata=_evaluation_metadata(row)))
    report=summarize_chunk_predictions(records,threshold);report['input_ineligible_pairs']=invalid
    report['calls']=len(timings)
    # First call reported separately; no fake warmup or repeated safety decision.
    report['latency_s']=dict(total_model_call_seconds=sum(forwards),total_boundary_seconds=sum(timings),first_call=timings[0] if timings else None,
        steady_calls=max(0,len(timings)-1),end_to_end_p50=quantile(timings[1:],.5),end_to_end_p95=quantile(timings[1:],.95),
        forward_p50=quantile(forwards[1:],.5),forward_p95=quantile(forwards[1:],.95),
        scope='Per boundary with all queries; end-to-end includes raw media decode, assembly, processor and model; first call excluded from steady quantiles')
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.with_suffix('.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    import argparse
    from .chunk_judge_model import load_judge
    from .predictor_judge_dataset import PredictorJudgeWindowDataset
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','checkpoint','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--device',default='cuda:0')
    p.add_argument('--threshold',type=float,default=.5);args=p.parse_args()
    data=PredictorJudgeWindowDataset(args.package,args.split)
    try:
        model=load_judge(args.checkpoint,device=args.device)
        if model.model_config.get('input_contract')!='chunk_start_v2':raise ValueError('Chunk checkpoint required')
        report=replay_episode_chunks(model,data,threshold=args.threshold,output=args.output)
        print(json.dumps(report,indent=2))
    finally:data.close()

if __name__=='__main__':main()
