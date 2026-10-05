"""Single-device/DDP training of the masked action-conditioned visual Predictor Judge."""
import argparse,copy,importlib.metadata,json,os,random
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from .predictor_judge_dataset import PredictorJudgeWindowDataset,collate_predictor_judge,PreparedPredictorJudgeCollator,file_hash
from .predictor_judge_eval import evaluate_predictor_judge
from .visual_train import publish_final,_rank_zero_call
from .visual_distributed import resolve_accumulation
from .predictor_judge_sampling import JudgeBatchSampler,consumed_draw_report,SamplingRows,diagnostic_validation_subset


def training_source_hashes():
    from . import predictor_judge_data,predictor_judge_schema,predictor_judge_dataset,predictor_judge_eval,predictor_judge_sampling,predictor_judge_cache,visual_cache,visual_distributed,visual_train,metrics,labels
    import jev.predictor_judge_model as model
    import jev.visual_training as training
    import jev.visual_model as visual
    files=[__file__]+[m.__file__ for m in (predictor_judge_data,predictor_judge_schema,predictor_judge_dataset,
           predictor_judge_eval,predictor_judge_sampling,predictor_judge_cache,visual_cache,visual_distributed,visual_train,metrics,labels,model,training,visual)]
    return {Path(p).name:file_hash(p) for p in files}


def run(config,output,*,device='cuda:0',resume=None,stop_after=None):
    from jev.predictor_judge_model import PredictorJudgeModel
    from jev.visual_training import fit_updates
    config=copy.deepcopy(config);world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if 'validation_samples' in config['evaluation']:
        raise ValueError('Use evaluation.validation_negative_samples: all validation positives are retained, this budget counts negatives only')
    if world>1 and not dist.is_initialized():
        local=int(os.environ['LOCAL_RANK'])
        if device.startswith('cuda'):torch.cuda.set_device(local);device=f'cuda:{local}'
        dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',timeout=timedelta(minutes=10))
    size=config['data']['batch_size']
    if config['training'].get('global_batch_size') is not None:
        config['training']['accumulation']=resolve_accumulation(config['training']['global_batch_size'],size,world)
    if config['model']['dtype'] not in ('float32','bfloat16'):raise ValueError('Use FP32 or BF16')
    output=Path(output).resolve()
    def create_output():
        if output.exists() and not resume:raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True)
        if (output/'final').exists():raise ValueError('Training has already finalized')
        (output/'sampling').mkdir(exist_ok=True)
    _rank_zero_call(create_output)
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    deterministic=config.get('deterministic',True)
    if type(deterministic) is not bool:raise ValueError('deterministic must be Boolean')
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'):raise ValueError('Invalid deterministic CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(deterministic)
    package=Path(config['data']['package']).resolve();frame_cache=config['data'].get('frame_cache')
    datasets={s:PredictorJudgeWindowDataset(package,s,**({'frame_cache':frame_cache} if frame_cache else {})) for s in ('train','validation','test')}
    meta=datasets['train'].summary
    def verify_sources():
        for res in meta['resources'].values():
            if file_hash(package/res['raw_root']/'record.json')!=res['record_sha256']:raise ValueError('Raw source identity changed')
            if file_hash(package/res['media_manifest'])!=res['media_manifest_sha256']:raise ValueError('Media manifest identity changed')
    _rank_zero_call(verify_sources)
    workers=config['data'].get('num_workers',0);sampling=dict(config['data'].get('sampling',{}))
    if set(sampling)-{'samples_per_epoch','positive_fraction'}:raise ValueError('Unsupported sampling settings')
    train_rows=SamplingRows(datasets['train'])
    model_cfg={**config['model'],'history_frames':meta['history_frames'],'max_actions':meta['max_actions'],'state_dim':len(meta['state_features'])}
    collator=PreparedPredictorJudgeCollator(model_cfg) if config['data'].get('prepare_in_workers') else collate_predictor_judge
    options={'num_workers':workers,'collate_fn':collator,'pin_memory':device.startswith('cuda')}
    if workers:options.update(persistent_workers=True,prefetch_factor=config['data'].get('prefetch_factor',2),multiprocessing_context='spawn')
    def training_loader(epoch,start):
        sampler=JudgeBatchSampler(train_rows,size,seed=seed,epoch=epoch,start_batch=start,rank=rank,world_size=world,**sampling)
        if rank==0:(output/'sampling'/f'epoch-{epoch:04d}.json').write_text(json.dumps(sampler.report(),indent=2)+'\n')
        return DataLoader(datasets['train'],batch_sampler=sampler,**options,generator=torch.Generator().manual_seed(seed+epoch+100000+rank))
    limit=config['evaluation'].get('max_batches')
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('max_batches must be positive')
    negative_budget=config['evaluation'].get('validation_negative_samples')
    validation_indices=None;validation_sampling={'selection':'full split','full_split':True}
    if negative_budget is not None:
        def select_validation():
            indices,report=diagnostic_validation_subset(SamplingRows(datasets['validation']),negative_samples=negative_budget,seed=seed+200001)
            (output/'validation-subset.json').write_text(json.dumps({**report,'row_indices':indices,
                'split_sha256':meta['file_sha256']['validation']},indent=2)+'\n')
            return indices,report
        validation_indices,validation_sampling=_rank_zero_call(select_validation)
    def eval_loader(split,diagnostic=False):
        data=datasets[split]
        indices=validation_indices if diagnostic and validation_indices is not None else range(len(data))
        indices=indices[rank::world]
        if limit is not None:indices=indices[:limit*size]
        return DataLoader(Subset(data,indices),batch_size=size,shuffle=False,**options,generator=torch.Generator().manual_seed(seed+200000+rank))
    loaders={s:eval_loader(s,diagnostic=s=='validation') for s in ('validation','test')}
    model=PredictorJudgeModel.from_pretrained(**model_cfg,device=device);model.set_normalization(**meta['normalization'])
    model.model_config['state_features']=meta['state_features'];model.model_config['normalization_source']='training groups only'
    params={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'vision_trainable':sum(p.numel() for p in model.backbone.visual.parameters() if p.requires_grad)}
    if params['vision_trainable']:raise ValueError('Vision encoder must remain frozen')
    identity={'package_sha256':file_hash(package/'dataset_metadata.json'),'data':config['data'],'evaluation':config['evaluation'],
              'seed':seed,'world_size':world,'deterministic':deterministic,'sources':training_source_hashes(),
              'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'runtime':{name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','peft','numpy','av')}}
    if rank==0:(output/'run.json').write_text(json.dumps({'config':config,'identity':identity,'device':device,'parameters':params},indent=2)+'\n')
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    def validation(current,step):
        report=evaluate_predictor_judge(current,loaders['validation'],output=output/'evaluations'/f'validation-{step:08d}.jsonl')
        report['sampling']={**validation_sampling,'max_batches_per_rank':limit}
        if rank==0:(output/'evaluations'/f'validation-{step:08d}.json').write_text(json.dumps(report,indent=2)+'\n')
        return report['nll']
    if world>1:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[int(os.environ['LOCAL_RANK'])] if device.startswith('cuda') else None,broadcast_buffers=False)
    trained=fit_updates(model,training_loader,config['training'],output,identity=identity,validation_fn=validation,resume=resume,stop_after=stop_after)
    def consumed():
        report=consumed_draw_report(train_rows,batch_size=size,seed=seed,epoch=trained['epoch'],batch_offset=trained['batch_offset'],world_size=world,**sampling)
        (output/'sampling-consumed.json').write_text(json.dumps(report,indent=2)+'\n');return report
    sampling_report=_rank_zero_call(consumed)
    if trained['status']!='completed':return trained
    del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    model=PredictorJudgeModel.load(Path(trained['best_checkpoint'])/'model',device=device)
    validation_report=evaluate_predictor_judge(model,eval_loader('validation'),output=output/'evaluations/final-validation.jsonl')
    test=evaluate_predictor_judge(model,loaders['test'],output=output/'evaluations/final-test.jsonl')
    def finalize(stage):
        import shutil
        shutil.copy2(output/'evaluations/final-test.jsonl',stage/'test.jsonl');model.save(stage/'model')
        report={'training':trained,'parameters':params,'sampling':sampling_report,'validation':validation_report,'test':test,
                'checkpoint_selection':{'metric':'diagnostic_validation_nll','sampling':validation_sampling},
                'data_counts':meta['counts'],'label_reasons':meta['label_reasons'],'identity':identity,
                'target':'new constraint violation during the actual remaining command suffix','checkpoint':str(output/'final/model'),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                'evaluation_capped':limit is not None,'scope':config.get('purpose','predictor judge training')}
        (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report
    return _rank_zero_call(lambda:publish_final(output,finalize))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume');p.add_argument('--stop-after-step','--stop-after',dest='stop_after',type=int)
    p.add_argument('--batch-size',type=int);p.add_argument('--global-batch-size',type=int);p.add_argument('--workers',type=int);p.add_argument('--frame-cache')
    a=p.parse_args(argv);config=json.loads(Path(a.config).read_text())
    for key,value in [('batch_size',a.batch_size),('num_workers',a.workers),('frame_cache',a.frame_cache)]:
        if value is not None:config['data'][key]=value
    if a.global_batch_size is not None:config['training']['global_batch_size']=a.global_batch_size
    try:
        result=run(config,a.output,device=a.device,resume=a.resume,stop_after=a.stop_after)
        if int(os.environ.get('RANK','0'))==0:print(json.dumps(result,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
