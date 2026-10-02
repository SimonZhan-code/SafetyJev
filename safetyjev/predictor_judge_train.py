"""Training entrypoint for the masked, action-conditioned visual Noul predictor judge."""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .predictor_judge_dataset import PredictorJudgeWindowDataset,collate_predictor_judge,file_hash
from .predictor_judge_eval import evaluate_predictor_judge
from .visual_train import publish_final
from .predictor_judge_sampling import JudgeBatchSampler,consumed_draw_report,SamplingRows


def training_source_hashes():
    from . import predictor_judge_data,predictor_judge_schema,predictor_judge_dataset,predictor_judge_eval,predictor_judge_sampling,metrics,labels
    import jev.predictor_judge_model as model
    import jev.visual_training as training
    import jev.visual_model as visual
    files=[__file__]+[m.__file__ for m in (predictor_judge_data,predictor_judge_schema,predictor_judge_dataset,
           predictor_judge_eval,predictor_judge_sampling,metrics,labels,model,training,visual)]
    return {Path(p).name:file_hash(p) for p in files}


def run(config,output,*,device='cuda:0',resume=None,stop_after=None):
    from jev.predictor_judge_model import PredictorJudgeModel
    from jev.visual_training import fit_updates
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Single-device reference trainer')
    if config['model']['dtype'] not in ('float32','bfloat16'):raise ValueError('Use FP32 or BF16')
    output=Path(output).resolve()
    if output.exists() and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final').exists():raise ValueError('Training has already finalized')
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    deterministic=config.get('deterministic',True)
    if deterministic:os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    package=Path(config['data']['package']).resolve()
    datasets={s:PredictorJudgeWindowDataset(package,s) for s in ('train','validation','test')};meta=datasets['train'].summary
    for res in meta['resources'].values():
        if file_hash(package/res['raw_root']/'record.json')!=res['record_sha256']:raise ValueError('Raw source identity changed')
    workers=config['data'].get('num_workers',0);size=config['data']['batch_size']
    sampling=dict(config['data'].get('sampling',{}))
    if set(sampling)-{'samples_per_epoch','positive_fraction'}:raise ValueError('Unsupported sampling settings')
    train_rows=SamplingRows(datasets['train'])
    (output/'sampling').mkdir(exist_ok=True)
    def training_loader(epoch,start):
        sampler=JudgeBatchSampler(train_rows,size,seed=seed,epoch=epoch,start_batch=start,**sampling)
        (output/'sampling'/f'epoch-{epoch:04d}.json').write_text(json.dumps(sampler.report(),indent=2)+'\n')
        return DataLoader(datasets['train'],batch_sampler=sampler,num_workers=workers,collate_fn=collate_predictor_judge,
                          generator=torch.Generator().manual_seed(seed+epoch+100000))
    loaders={s:DataLoader(datasets[s],batch_size=size,shuffle=False,num_workers=workers,collate_fn=collate_predictor_judge,
                          generator=torch.Generator().manual_seed(seed+200000)) for s in ('validation','test')}
    model_cfg={**config['model'],'history_frames':meta['history_frames'],'max_actions':meta['max_actions'],'state_dim':len(meta['state_features'])}
    model=PredictorJudgeModel.from_pretrained(**model_cfg,device=device);model.set_normalization(**meta['normalization'])
    model.model_config['state_features']=meta['state_features'];model.model_config['normalization_source']='training groups only'
    identity={'package_sha256':file_hash(package/'dataset_metadata.json'),'data':config['data'],'evaluation':config['evaluation'],
              'seed':seed,'deterministic':deterministic,'sources':training_source_hashes()}
    (output/'run.json').write_text(json.dumps({'config':config,'identity':identity,'device':device},indent=2)+'\n')
    limit=config['evaluation'].get('max_batches')
    def validation(current,step):
        report=evaluate_predictor_judge(current,loaders['validation'],max_batches=limit,output=output/'evaluations'/f'validation-{step:08d}.jsonl')
        (output/'evaluations'/f'validation-{step:08d}.json').write_text(json.dumps(report,indent=2)+'\n');return report['nll']
    trained=fit_updates(model,training_loader,config['training'],output,identity=identity,validation_fn=validation,resume=resume,stop_after=stop_after)
    sampling_report=consumed_draw_report(train_rows,batch_size=size,seed=seed,epoch=trained['epoch'],
                                         batch_offset=trained['batch_offset'],**sampling)
    (output/'sampling-consumed.json').write_text(json.dumps(sampling_report,indent=2)+'\n')
    if trained['status']!='completed':return trained
    del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    model=PredictorJudgeModel.load(Path(trained['best_checkpoint'])/'model',device=device)
    def finalize(stage):
        test=evaluate_predictor_judge(model,loaders['test'],max_batches=limit,output=stage/'test.jsonl')
        model.save(stage/'model')
        report={'training':trained,'sampling':sampling_report,'test':test,'data_counts':meta['counts'],'label_reasons':meta['label_reasons'],
                'identity':identity,'target':'new constraint violation during the actual remaining command suffix',
                'checkpoint':str(output/'final/model'),'scope':config.get('purpose','predictor judge training')}
        (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report
    return publish_final(output,finalize)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    a=p.parse_args(argv);result=run(json.loads(Path(a.config).read_text()),a.output,device=a.device,resume=a.resume,stop_after=a.stop_after)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
