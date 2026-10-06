"""Download pinned benchmark variants; reuse the previously verified policy steps."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parents[2]
config=json.loads((ROOT/'configs/domain-sweep-policies.json').read_text())
info=HfApi().dataset_info('IDEAS-Lab-Northwestern/ManiGuard-Bench',revision=config['benchmark_revision'])
files=[s.rfilename for s in info.siblings]
for family,spec in config['families'].items():
    pipeline=spec['pipeline'];spec['scenes_by_level']={}
    for level in ['base','target','language','location','env']:
        spec['scenes_by_level'][level]=sorted({'/'.join(f.split('/')[1:3]) for f in files if f.startswith(pipeline+'/') and '/'+level+'/' in f})
    spec['repo']=f'IDEAS-Lab-Northwestern/pi05-base-datagen-v1-{family}-joint-2cam-lora'
    print('DOWNLOAD',family,{l:len(v) for l,v in spec['scenes_by_level'].items()},flush=True)
    snapshot_download('IDEAS-Lab-Northwestern/ManiGuard-Bench',repo_type='dataset',revision=config['benchmark_revision'],allow_patterns=[pipeline+'/task_*/*/*'],local_dir='/workspace/data/maniguard-bench',max_workers=8)
    checkpoint=Path('/workspace/checkpoints/pi05-'+family)/spec['step']
    if not checkpoint.is_dir():raise ValueError('Required pinned policy step not available: '+str(checkpoint))
    spec['ready']=True
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/domain-sweep-resources.json').write_text(json.dumps(config,indent=2)+'\n')
    print('READY',family,flush=True)
