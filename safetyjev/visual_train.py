"""Jar/multi-family launch layer for the fork's current-camera Noul trainer."""
import copy
from datetime import timedelta
import torch.distributed as dist
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .visual_dataset import APWindowDataset, collate_torch, PreparedVisualCollator
from .visual_distributed import GlobalBatchSampler, resolve_accumulation
from .data_preparation import _local_path


class EpochBatchSampler:
    def __init__(self, size, batch_size, *, seed, epoch, start_batch=0, weights=None):
        if size < 1 or batch_size < 1 or start_batch < 0:
            raise ValueError("Invalid deterministic sampler arguments")
        self.size,self.batch_size,self.seed,self.epoch,self.start_batch,self.weights = size,batch_size,seed,epoch,start_batch,weights

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        if self.weights is None:
            order=torch.randperm(self.size,generator=generator).tolist()
        else:
            order=torch.multinomial(torch.as_tensor(self.weights,dtype=torch.double),self.size,replacement=True,generator=generator).tolist()
        for start in range(self.start_batch*self.batch_size,self.size,self.batch_size):
            yield order[start:start+self.batch_size]

    def __len__(self):
        return max(0,math.ceil(self.size/self.batch_size)-self.start_batch)


def confusion(targets, yes_probabilities, threshold=.5):
    result={"tn":0,"fp":0,"fn":0,"tp":0}
    for target,p in zip(targets,yes_probabilities):
        truth=target[1] == 1; predicted=p >= threshold
        result[("t" if truth==predicted else "f")+("p" if predicted else "n")]+=1
    result["yes_precision"]=result["tp"]/(result["tp"]+result["fp"]) if result["tp"]+result["fp"] else None
    result["yes_recall"]=result["tp"]/(result["tp"]+result["fn"]) if result["tp"]+result["fn"] else None
    return result



def answer_metrics(targets, probabilities, threshold=.5):
    counts=confusion(targets,probabilities,threshold)
    result={}
    for name,tp,fp,fn in [('yes',counts['tp'],counts['fp'],counts['fn']),('no',counts['tn'],counts['fn'],counts['fp'])]:
        support=tp+fn
        result[name]={'support':support,'precision':tp/(tp+fp) if tp+fp else None,
                      'recall':tp/support if support else None,
                      'f1':2*tp/(2*tp+fp+fn) if support else None}
    f1=[r['f1'] for r in result.values() if r['f1'] is not None]
    recall=[r['recall'] for r in result.values() if r['recall'] is not None]
    result['macro_f1']=sum(f1)/len(f1) if f1 else None
    result['balanced_accuracy']=sum(recall)/2 if len(recall)==2 else None
    return result

def _hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def verify_package(package, *, verify_raw=True):
    package=Path(package).resolve()
    summary=json.loads((package/"dataset_metadata.json").read_text())
    for split,digest in summary["file_sha256"].items():
        if _hash(package/(split+".jsonl"))!=digest:raise ValueError("Dataset split bytes changed: "+split)
    if summary.get('method')=='semantic_safety':
        from .predictor_judge_dataset import verify_semantic_resource,resource_record_path
        from .source_episodes import MediaReader
        if _hash(package/'semantic_definitions.json')!=summary['semantic_definitions_sha256']:
            raise ValueError('Semantic definitions changed')
        for res in summary['resources'].values():
            verify_semantic_resource(package,res)
            if _hash(resource_record_path(package,res))!=res['record_sha256']:raise ValueError('Source record changed')
            path=(package/res['media_manifest']).resolve()
            if not path.is_relative_to(package) or _hash(path)!=res['media_manifest_sha256']:
                raise ValueError('Source image manifest changed')
            if verify_raw:
                with MediaReader(package/res['raw_root']) as reader:
                    for ref,digest in json.loads(path.read_text()).items():reader.read(ref,digest)
        return summary
    for resource in summary["resources"].values():
        raw=(package/resource["raw_root"]).resolve();manifest=raw/"manifest.jsonl"
        if _hash(manifest)!=resource["raw_manifest_sha256"]:raise ValueError("Raw manifest changed")
        if not verify_raw:continue
        with manifest.open() as stream:
            for line in stream:
                for item in json.loads(line)["files"]:
                    path=_local_path(raw,item["path"])
                    if not path.is_file() or path.stat().st_size!=item["bytes"] or _hash(path)!=item["sha256"]:
                        raise ValueError("Raw observation or label bytes changed: "+item["path"])
    return summary


def publish_final(output, build):
    output=Path(output);final=output/"final"
    if final.exists():raise FileExistsError(final)
    stage=Path(tempfile.mkdtemp(prefix=".final-writing-",dir=output))
    try:
        result=build(stage)
        os.replace(stage,final)
        return result
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def _semantic_metrics(targets, probabilities, metadata):
    """Group semantic binary supervision; metadata never enters model.forward."""
    from .metrics import binary_metrics
    if not len(targets) == len(probabilities) == len(metadata):
        raise ValueError('Evaluation metadata and predictions must align')
    groups = {'by_semantic':defaultdict(list), 'by_family':defaultdict(list)}
    for target, probability, meta in zip(targets, probabilities, metadata):
        if not meta.get('semantic_id'):
            continue
        if target not in ([1.,0.], [0.,1.]):
            raise ValueError('Semantic evaluation requires binary No/Yes targets')
        pair = (int(target[1]), probability[1])
        groups['by_semantic'][meta['semantic_id']].append(pair)
        if meta.get('family'):
            groups['by_family'][meta['family']].append(pair)
    if not groups['by_semantic']:
        return {}
    return {key:{name:binary_metrics(pairs,.5) for name,pairs in values.items()}
            for key,values in groups.items()}


@torch.inference_mode()
def _evaluate_visual_local(model, loader, *, temperature=1., max_batches=None, output=None):
    from jev.metrics import evaluate_probabilities, softmax
    model.eval();logits_all=[];targets_all=[];ids=[];queries=[];metadata=[];elapsed=0.
    for index,batch in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        if model.head.weight.device.type=="cuda":torch.cuda.synchronize(model.head.weight.device)
        started=time.perf_counter();logits=model(**batch["inputs"])
        if model.head.weight.device.type=="cuda":torch.cuda.synchronize(model.head.weight.device)
        elapsed+=time.perf_counter()-started
        if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite evaluation logits")
        logits_all.extend(logits.float().cpu().tolist());targets_all.extend(batch["targets"].tolist())
        ids.extend(batch["sample_ids"]);queries.extend(batch["query_ids"])
        metadata.extend(batch.get('metadata', [{} for _ in batch['sample_ids']]))
    if not logits_all:raise ValueError("Evaluation split produced no observations")
    probabilities=[softmax(values,temperature) for values in logits_all]
    metrics=evaluate_probabilities(targets_all,probabilities)
    metrics["confusion"]=confusion(targets_all,[p[1] for p in probabilities])
    metrics['answers']=answer_metrics(targets_all,[p[1] for p in probabilities])
    metrics.update(_semantic_metrics(targets_all,probabilities,metadata))
    groups=defaultdict(list)
    for i,query in enumerate(queries):groups[query].append(i)
    metrics["by_query"]={q:{**evaluate_probabilities([targets_all[i] for i in ix],[probabilities[i] for i in ix]),
                             "confusion":confusion([targets_all[i] for i in ix],[probabilities[i][1] for i in ix]),
                             'answers':answer_metrics([targets_all[i] for i in ix],[probabilities[i][1] for i in ix])}
                          for q,ix in groups.items()}
    metrics.update(evaluated=len(ids),available=len(loader.dataset),temperature=temperature,
                   model_time_seconds=elapsed,time_scope="forward including host-to-device transfer; excludes loader and worker preprocessing")
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("w") as stream:
            for i in range(len(ids)):
                stream.write(json.dumps({"id":ids[i],"query_id":queries[i],"logits":logits_all[i],
                                         "target":targets_all[i],"probabilities":probabilities[i],
                                         "metadata":metadata[i]},allow_nan=False)+"\n")
    return metrics,logits_all,targets_all



def _rank_zero_call(fn):
    distributed=dist.is_initialized();rank=dist.get_rank() if distributed else 0
    result=[None,None]
    if rank==0:
        try:result[0]=fn()
        except Exception as exc:result[1]=f"{type(exc).__name__}: {exc}"
    if distributed:dist.broadcast_object_list(result,src=0)
    if result[1]:raise RuntimeError(result[1])
    return result[0]


@torch.inference_mode()
def evaluate_visual(model, loader, *, temperature=1., max_batches=None, output=None):
    if not dist.is_initialized():
        return _evaluate_visual_local(model,loader,temperature=temperature,max_batches=max_batches,output=output)
    if output is None:raise ValueError("Distributed evaluation requires a shared prediction output path")
    rank=dist.get_rank();world=dist.get_world_size();output=Path(output)
    part=output.with_name(output.name+f".rank{rank}")
    if len(loader.dataset):
        metrics,_,_=_evaluate_visual_local(model,loader,temperature=temperature,max_batches=max_batches,output=part)
        elapsed=metrics['model_time_seconds']
    else:
        part.parent.mkdir(parents=True,exist_ok=True);part.write_text('');elapsed=0.
    timings=[None]*world;dist.all_gather_object(timings,elapsed)
    def merge():
        from jev.metrics import evaluate_probabilities
        targets=[];probs=[];metadata=[];groups=defaultdict(list);seen=set()
        temporary=output.with_suffix(output.suffix+'.tmp')
        with temporary.open('w') as writer:
            for r in range(world):
                with output.with_name(output.name+f".rank{r}").open() as stream:
                    for line in stream:
                        row=json.loads(line)
                        if row['id'] in seen:raise ValueError("Duplicate distributed evaluation sample")
                        seen.add(row['id']);groups[row['query_id']].append(len(targets))
                        targets.append(row['target']);probs.append(row['probabilities']);writer.write(line)
                        metadata.append(row.get('metadata',{}))
        if not targets:raise ValueError("Evaluation has no samples")
        result=evaluate_probabilities(targets,probs)
        result['confusion']=confusion(targets,[p[1] for p in probs])
        result['answers']=answer_metrics(targets,[p[1] for p in probs])
        result.update(_semantic_metrics(targets,probs,metadata))
        result['by_query']={q:{**evaluate_probabilities([targets[i] for i in ix],[probs[i] for i in ix]),
                            'confusion':confusion([targets[i] for i in ix],[probs[i][1] for i in ix]),
                            'answers':answer_metrics([targets[i] for i in ix],[probs[i][1] for i in ix])} for q,ix in groups.items()}
        result.update(evaluated=len(targets),temperature=temperature,model_time_seconds=max(timings),
                      time_scope='max rank forward including host-to-device transfer; excludes loader and worker preprocessing')
        os.replace(temporary,output)
        for r in range(world):output.with_name(output.name+f".rank{r}").unlink()
        return result
    return _rank_zero_call(merge),[],[]

def run(config, output, *, device="cuda:0", resume=None, stop_after=None):
    from .experiment_tracking import ExperimentTracker
    tracker = ExperimentTracker(output, resume=resume)
    status = 'failed'
    try:
        result = _run(config, output, device=device, resume=resume, stop_after=stop_after, tracker=tracker)
        status = result.get('status', result.get('training', {}).get('status', 'completed'))
        return result
    finally:
        tracker.finish(status)


def _run(config, output, *, device, resume, stop_after, tracker):
    from jev.visual_model import VisualDecisionModel
    from jev.visual_training import fit_updates
    config=copy.deepcopy(config)
    run_test=config["evaluation"].get("run_test", True)
    if type(run_test) is not bool:raise ValueError("evaluation.run_test must be Boolean")
    world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
    if world>1 and not dist.is_initialized():
        local=int(os.environ['LOCAL_RANK'])
        if device.startswith('cuda'):torch.cuda.set_device(local);device=f'cuda:{local}'
        dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',timeout=timedelta(minutes=10))
    if config['training'].get('global_batch_size') is not None:
        config['training']['accumulation']=resolve_accumulation(config['training']['global_batch_size'],config['data']['batch_size'],world)
    if config["model"]["dtype"] not in ("float32","bfloat16"):
        raise ValueError("The training reference uses float32 or bfloat16, without FP16 loss scaling")
    deterministic=config.get("deterministic",True)
    if type(deterministic) is not bool:raise ValueError("deterministic must be Boolean")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8",":16:8"):
            raise ValueError("Set a deterministic CUBLAS_WORKSPACE_CONFIG before training")
    torch.use_deterministic_algorithms(deterministic)
    package=Path(config["data"]["package"]).resolve()
    frame_cache=config['data'].get('frame_cache')
    def preflight():
        if frame_cache:
            from .visual_cache import FrameCache
            FrameCache(frame_cache,package).close()
            return verify_package(package,verify_raw=False)
        return verify_package(package)
    summary=_rank_zero_call(preflight)
    if summary["window"]["history_frames"]!=1:
        raise ValueError("First-round visual training requires current-frame H1 data")
    output=Path(output).resolve()
    def create_output():
        if output.exists() and not resume:raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True)
        if (output/"final/report.json").exists():raise ValueError("This run has already finished")
    _rank_zero_call(create_output)
    _rank_zero_call(lambda:tracker.start(config))
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith("cuda"):torch.cuda.manual_seed_all(seed)
    datasets={split:APWindowDataset(package,split,**({'frame_cache':frame_cache} if frame_cache else {})) for split in (("train","validation","test") if run_test else ("train","validation"))}
    workers=config["data"]["num_workers"];batch_size=config["data"]["batch_size"]
    balance=config["data"].get("balance","uniform")
    weights=None if balance=="uniform" else datasets["train"].training_weights(balance)
    collator=PreparedVisualCollator(config['model']) if config['data'].get('prepare_in_workers',False) else collate_torch
    loader_options={'num_workers':workers,'collate_fn':collator,'pin_memory':device.startswith('cuda')}
    if workers:
        loader_options.update(persistent_workers=True,prefetch_factor=config['data'].get('prefetch_factor',2),multiprocessing_context='spawn')
    def train_loader(epoch,start_batch):
        sampler=GlobalBatchSampler(len(datasets['train']),batch_size,seed=seed,epoch=epoch,rank=rank,world_size=world,
                                   start_batch=start_batch,weights=weights)
        return DataLoader(datasets['train'],batch_sampler=sampler,**loader_options,
                          generator=torch.Generator().manual_seed(seed+epoch+100000+rank))
    def eval_loader(split,limit=None):
        dataset=datasets[split]
        indices=list(range(len(dataset)))
        if limit is not None and limit<len(indices):
            g=torch.Generator().manual_seed(seed+200001)
            indices=torch.randperm(len(dataset),generator=g)[:limit].sort().values.tolist()
        if split=='validation' and limit is not None and rank==0:
            (output/'validation-subset.json').write_text(json.dumps({'row_indices':indices,'split_sha256':summary['file_sha256'].get('validation'),
                'selection':'uniform without replacement, fixed seed; diagnostic subset'})+'\n')
        indices=indices[rank::world]
        # Cap before worker prefetch starts, so smoke evaluation exhausts its
        # loader instead of discarding an active pinned-memory worker pipeline.
        batch_limit=config['evaluation'].get('max_batches')
        if batch_limit is not None:
            if type(batch_limit) is not int or batch_limit<1:raise ValueError('max_batches must be positive')
            indices=indices[:batch_limit*batch_size]
        from torch.utils.data import Subset
        return DataLoader(Subset(dataset,indices),batch_size=batch_size,**loader_options,shuffle=False,
                          generator=torch.Generator().manual_seed(seed+200000+rank))
    eval_loaders={s:eval_loader(s,config['evaluation'].get('validation_samples') if s=='validation' else None)
                  for s in (('validation','test') if run_test else ('validation',))}
    import jev.visual_model as vm
    import jev.visual_training as vt
    from . import visual_dataset,visual_distributed,visual_cache
    source_hashes={str(Path(p).name):_hash(p) for p in (__file__,vm.__file__,vt.__file__,visual_dataset.__file__,visual_distributed.__file__,visual_cache.__file__)}
    identity={"package_sha256":_hash(package/"dataset_metadata.json"),"split_hashes":summary["file_sha256"],
              "sources":source_hashes,"seed":seed,"world_size":world,"frame_cache":bool(frame_cache),"batch_size":batch_size,"balance":balance,
              "num_workers":workers,"evaluation":config["evaluation"],"deterministic":deterministic,
              "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "runtime":{name:importlib.metadata.version(name) for name in
                         ("torch","torchvision","transformers","peft","numpy","av")}}
    model=VisualDecisionModel.from_pretrained(**config["model"],device=device)
    params={"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vision_trainable":sum(p.numel() for p in model.backbone.visual.parameters() if p.requires_grad)}
    if params["vision_trainable"]:raise ValueError("The initial visual experiment must keep the vision encoder frozen")
    if rank==0:(output/"run.json").write_text(json.dumps({"config":config,"identity":identity,"device":device,"parameters":params},indent=2)+"\n")
    if device.startswith("cuda"):torch.cuda.reset_peak_memory_stats(device)
    max_batches=config["evaluation"].get("max_batches")
    def validation(current,step):
        metrics,_,_=evaluate_visual(current,eval_loaders["validation"],max_batches=max_batches,
                                    output=output/"evaluations"/f"validation-{step:08d}.jsonl")
        if rank==0:(output/"evaluations"/f"validation-{step:08d}.json").write_text(json.dumps(metrics,indent=2)+"\n")
        tracker.evaluation(metrics,step)
        return metrics["nll"]
    if world>1:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[int(os.environ['LOCAL_RANK'])] if device.startswith('cuda') else None,
                                                        broadcast_buffers=False)
    trained=fit_updates(model,train_loader,config["training"],output,identity=identity,
                        validation_fn=validation,resume=resume,stop_after=stop_after,step_fn=tracker.step)
    if trained["status"]!="completed":return trained
    del model
    if device.startswith("cuda"):torch.cuda.empty_cache()
    model=VisualDecisionModel.load(Path(trained["best_checkpoint"])/"model",device=device)
    temperature=1.
    model.set_temperature(temperature)
    full_validation,_,_=evaluate_visual(model,eval_loader('validation'),output=output/'evaluations/final-validation.jsonl',max_batches=max_batches)
    test=None
    if run_test:
        test,_,_=evaluate_visual(model,eval_loaders["test"],temperature=temperature,max_batches=max_batches,output=output/'evaluations/final-test.jsonl')
        tracker.evaluation(test,trained['completed_step'],'test')
    tracker.evaluation(full_validation,trained['completed_step'],'final_validation')
    def finalize(stage):
        if run_test:shutil.copy2(output/'evaluations/final-test.jsonl',stage/'test.jsonl')
        model.save(stage/"model")
        report={"training":trained,"parameters":params,"identity":identity,
                "temperature":temperature,"validation":full_validation,"test":test,
                "test_status":"evaluated" if run_test else "not_evaluated",
                "checkpoint":str(output/"final/model"),"target":"current-question yes/no, not action-conditioned forecasting",
                "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
        (stage/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        return report
    return _rank_zero_call(lambda:publish_final(output,finalize))


def main(argv=None):
    parser=argparse.ArgumentParser(description="Train current-dual-camera visual Open-Jev")
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--resume")
    parser.add_argument("--stop-after-step",type=int)
    parser.add_argument('--batch-size',type=int,help='Per-device microbatch')
    parser.add_argument('--global-batch-size',type=int,help='Samples per optimizer update; accumulation is derived')
    parser.add_argument('--workers',type=int);parser.add_argument('--frame-cache')
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    if args.batch_size is not None:config['data']['batch_size']=args.batch_size
    if args.global_batch_size is not None:config['training']['global_batch_size']=args.global_batch_size
    if args.workers is not None:config['data']['num_workers']=args.workers
    if args.frame_cache is not None:config['data']['frame_cache']=args.frame_cache
    try:
        result=run(config,args.output,device=args.device,resume=args.resume,stop_after=args.stop_after_step)
        if int(os.environ.get('RANK','0'))==0:print(json.dumps(result,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=="__main__":main()
