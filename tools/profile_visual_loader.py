"""Measure the actual training input path without allocating a model or GPU."""
import argparse,json,time
import numpy as np
from torch.utils.data import DataLoader
from safetyjev.visual_dataset import APWindowDataset,PreparedVisualCollator,collate_torch
from safetyjev.visual_distributed import GlobalBatchSampler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--batches',type=int,default=100)
    p.add_argument('--output',required=True);args=p.parse_args()
    if args.batches<2:raise ValueError('Use at least two batches')
    config=json.load(open(args.config));data=config['data']
    task=config.get('task','classifier')
    if task=='predictor_judge':
        from safetyjev.predictor_judge_dataset import PredictorJudgeWindowDataset,PreparedPredictorJudgeCollator,collate_predictor_judge
        from safetyjev.predictor_judge_sampling import JudgeBatchSampler,SamplingRows
        dataset=PredictorJudgeWindowDataset(data['package'],'train',frame_cache=data.get('frame_cache'))
        sampler=JudgeBatchSampler(SamplingRows(dataset),data['batch_size'],seed=config['seed'],epoch=0,**data.get('sampling',{}))
        model_config={**config['model'],**{k:dataset.summary[k] for k in ['history_frames','max_actions']}}
        collator=PreparedPredictorJudgeCollator(model_config) if data.get('prepare_in_workers') else collate_predictor_judge
    elif task=='classifier':
        dataset=APWindowDataset(data['package'],'train',frame_cache=data.get('frame_cache'))
        balance=data.get('balance','uniform')
        weights=None if balance=='uniform' else dataset.training_weights(balance)
        sampler=GlobalBatchSampler(len(dataset),data['batch_size'],seed=config['seed'],epoch=0,weights=weights)
        collator=PreparedVisualCollator(config['model']) if data.get('prepare_in_workers') else collate_torch
    else:raise ValueError('Unknown training task')
    options=dict(num_workers=args.workers,collate_fn=collator)
    if args.workers:options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(dataset,batch_sampler=sampler,**options)
    start=time.perf_counter();iterator=iter(loader);waits=[];samples=0
    for index in range(min(args.batches,len(loader))):
        before=time.perf_counter();batch=next(iterator);waits.append(time.perf_counter()-before);samples+=len(batch['targets'])
    elapsed=time.perf_counter()-start
    result={'samples':samples,'seconds_including_startup':elapsed,'samples_per_second':samples/elapsed,
            'first_batch_wait_s':waits[0],'batch_wait_p50_s':float(np.quantile(waits[1:],.5)) if len(waits)>1 else None,
            'batch_wait_p95_s':float(np.quantile(waits[1:],.95)) if len(waits)>1 else None,
            'workers':args.workers,'scope':'loader-only single consumer; does not establish multi-GPU training throughput'}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
