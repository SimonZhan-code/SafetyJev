"""Jar/multi-family launch layer for the fork's current-camera Noul trainer."""
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

from .visual_dataset import APWindowDataset, collate_torch
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


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def verify_package(package):
    package=Path(package).resolve()
    summary=json.loads((package/"dataset_metadata.json").read_text())
    for split,digest in summary["file_sha256"].items():
        if _hash(package/(split+".jsonl"))!=digest:raise ValueError("Dataset split bytes changed: "+split)
    for resource in summary["resources"].values():
        raw=(package/resource["raw_root"]).resolve();manifest=raw/"manifest.jsonl"
        if _hash(manifest)!=resource["raw_manifest_sha256"]:raise ValueError("Raw manifest changed")
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


@torch.inference_mode()
def evaluate_visual(model, loader, *, temperature=1., max_batches=None, output=None):
    from jev.metrics import evaluate_probabilities, softmax
    model.eval();logits_all=[];targets_all=[];ids=[];queries=[];elapsed=0.
    for index,batch in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        if model.head.weight.device.type=="cuda":torch.cuda.synchronize(model.head.weight.device)
        started=time.perf_counter();logits=model(**batch["inputs"])
        if model.head.weight.device.type=="cuda":torch.cuda.synchronize(model.head.weight.device)
        elapsed+=time.perf_counter()-started
        if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite evaluation logits")
        logits_all.extend(logits.float().cpu().tolist());targets_all.extend(batch["targets"].tolist())
        ids.extend(batch["sample_ids"]);queries.extend(batch["query_ids"])
    if not logits_all:raise ValueError("Evaluation split produced no observations")
    probabilities=[softmax(values,temperature) for values in logits_all]
    metrics=evaluate_probabilities(targets_all,probabilities)
    metrics["confusion"]=confusion(targets_all,[p[1] for p in probabilities])
    groups=defaultdict(list)
    for i,query in enumerate(queries):groups[query].append(i)
    metrics["by_query"]={q:{**evaluate_probabilities([targets_all[i] for i in ix],[probabilities[i] for i in ix]),
                             "confusion":confusion([targets_all[i] for i in ix],[probabilities[i][1] for i in ix])}
                          for q,ix in groups.items()}
    metrics.update(evaluated=len(ids),available=len(loader.dataset),temperature=temperature,
                   model_time_seconds=elapsed,time_scope="processor+model batches; excludes video decode and transport")
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("w") as stream:
            for i in range(len(ids)):
                stream.write(json.dumps({"id":ids[i],"query_id":queries[i],"logits":logits_all[i],
                                         "target":targets_all[i],"probabilities":probabilities[i]},allow_nan=False)+"\n")
    return metrics,logits_all,targets_all


def run(config, output, *, device="cuda:0", resume=None, stop_after=None):
    from jev.visual_model import VisualDecisionModel
    from jev.visual_training import fit_updates
    if int(os.environ.get("WORLD_SIZE","1"))!=1:
        raise ValueError("This reference trainer uses one device; do not launch duplicate torchrun processes")
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
    summary=verify_package(package)
    if summary["window"]["history_frames"]!=1:
        raise ValueError("First-round visual training requires current-frame H1 data")
    output=Path(output).resolve()
    if output.exists() and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"final/report.json").exists():raise ValueError("This run has already finished")
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith("cuda"):torch.cuda.manual_seed_all(seed)
    datasets={split:APWindowDataset(package,split) for split in ("train","validation","test")}
    workers=config["data"]["num_workers"];batch_size=config["data"]["batch_size"]
    balance=config["data"].get("balance","uniform")
    weights=None if balance=="uniform" else datasets["train"].training_weights(balance)
    def train_loader(epoch,start_batch):
        sampler=EpochBatchSampler(len(datasets["train"]),batch_size,seed=seed,epoch=epoch,start_batch=start_batch,weights=weights)
        return DataLoader(datasets["train"],batch_sampler=sampler,num_workers=workers,collate_fn=collate_torch,
                          generator=torch.Generator().manual_seed(seed+epoch+100000))
    eval_loaders={s:DataLoader(datasets[s],batch_size=batch_size,num_workers=workers,collate_fn=collate_torch,
                              shuffle=False,generator=torch.Generator().manual_seed(seed+200000))
                  for s in ("validation","test")}
    import jev.visual_model as vm
    import jev.visual_training as vt
    from . import visual_dataset
    source_hashes={str(Path(p).name):_hash(p) for p in (__file__,vm.__file__,vt.__file__,visual_dataset.__file__)}
    identity={"package_sha256":_hash(package/"dataset_metadata.json"),"split_hashes":summary["file_sha256"],
              "sources":source_hashes,"seed":seed,"batch_size":batch_size,"balance":balance,
              "num_workers":workers,"evaluation":config["evaluation"],"deterministic":deterministic,
              "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "runtime":{name:importlib.metadata.version(name) for name in
                         ("torch","torchvision","transformers","peft","numpy","av")}}
    model=VisualDecisionModel.from_pretrained(**config["model"],device=device)
    params={"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vision_trainable":sum(p.numel() for p in model.backbone.visual.parameters() if p.requires_grad)}
    if params["vision_trainable"]:raise ValueError("The initial visual experiment must keep the vision encoder frozen")
    (output/"run.json").write_text(json.dumps({"config":config,"identity":identity,"device":device,"parameters":params},indent=2)+"\n")
    if device.startswith("cuda"):torch.cuda.reset_peak_memory_stats(device)
    max_batches=config["evaluation"].get("max_batches")
    def validation(current,step):
        metrics,_,_=evaluate_visual(current,eval_loaders["validation"],max_batches=max_batches,
                                    output=output/"evaluations"/f"validation-{step:08d}.jsonl")
        (output/"evaluations"/f"validation-{step:08d}.json").write_text(json.dumps(metrics,indent=2)+"\n")
        return metrics["nll"]
    trained=fit_updates(model,train_loader,config["training"],output,identity=identity,
                        validation_fn=validation,resume=resume,stop_after=stop_after)
    if trained["status"]!="completed":return trained
    del model
    if device.startswith("cuda"):torch.cuda.empty_cache()
    model=VisualDecisionModel.load(Path(trained["best_checkpoint"])/"model",device=device)
    def finalize(stage):
        temperature=1.
        model.set_temperature(temperature)
        test,_,_=evaluate_visual(model,eval_loaders["test"],temperature=temperature,max_batches=max_batches,
                                 output=stage/"test.jsonl")
        model.save(stage/"model")
        report={"training":trained,"parameters":params,"identity":identity,
                "temperature":temperature,"test":test,
                "checkpoint":str(output/"final/model"),"target":"current-question yes/no, not action-conditioned forecasting",
                "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
        (stage/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        return report
    return publish_final(output,finalize)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Train current-dual-camera visual Open-Jev")
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--resume")
    parser.add_argument("--stop-after-step",type=int)
    args=parser.parse_args(argv)
    result=run(json.loads(Path(args.config).read_text()),args.output,device=args.device,resume=args.resume,stop_after=args.stop_after_step)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
