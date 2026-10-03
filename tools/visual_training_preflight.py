"""Check a training host and report effective batch before loading the full model."""
import argparse,json,os,shutil
from pathlib import Path
import torch
from safetyjev.visual_distributed import resolve_accumulation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--gpus',type=int,default=1);args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    if not 1<=args.gpus<=torch.cuda.device_count():raise ValueError('Requested GPU count is not visible')
    micro=cfg['data']['batch_size']
    total=cfg['training'].get('global_batch_size')
    if total is None:total=micro*args.gpus*cfg['training']['accumulation']
    accumulation=resolve_accumulation(total,micro,args.gpus);devices=[]
    for i in range(args.gpus):
        with torch.cuda.device(i):
            x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16);y=x@x;torch.cuda.synchronize()
            if not torch.isfinite(y).all():raise ValueError('CUDA math smoke failed')
            props=torch.cuda.get_device_properties(i);devices.append({'name':props.name,'capability':torch.cuda.get_device_capability(i),'memory_GiB':props.total_memory/2**30})
    print(json.dumps({'devices':devices,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'compiled_arches':torch.cuda.get_arch_list(),
                     'microbatch':micro,'accumulation':accumulation,'global_batch':total,'free_disk_GiB':shutil.disk_usage('.').free/2**30,
                     'cpu_affinity_count':len(os.sched_getaffinity(0)),
                     'scope':'CUDA smoke and configuration only; full model memory and NCCL topology need target-host acceptance'},indent=2))
if __name__=='__main__':main()
