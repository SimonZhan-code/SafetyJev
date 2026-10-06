"""Download pinned fine-tuned pi05 weights and base-only non-Jar scenes."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
ROOT=Path('/workspace/SafetyJev')
config=json.loads((ROOT/'configs/base-sweep-policies.json').read_text())
api=HfApi()
info=api.dataset_info('IDEAS-Lab-Northwestern/ManiGuard-Bench',revision=config['benchmark_revision'])
files=[s.rfilename for s in info.siblings]
for family, spec in config['families'].items():
    pipeline=spec['pipeline']
    spec['scenes']=sorted({'/'.join(f.split('/')[1:3]) for f in files if f.startswith(pipeline+'/') and '/base/' in f})
    spec['repo']=f'IDEAS-Lab-Northwestern/pi05-base-datagen-v1-{family}-joint-2cam-lora'
    print('DOWNLOAD',family,len(spec['scenes']),'base scenes',flush=True)
    snapshot_download('IDEAS-Lab-Northwestern/ManiGuard-Bench',repo_type='dataset',revision=config['benchmark_revision'],allow_patterns=[pipeline+'/task_*/base/*'],local_dir='/workspace/data/maniguard-bench',max_workers=8)
    snapshot_download(spec['repo'],revision=spec['revision'],allow_patterns=[spec['step']+'/*','README.md'],local_dir='/workspace/checkpoints/pi05-'+family,max_workers=8)
    spec['ready']=True
    (ROOT/'artifacts/base-sweep-download.json').write_text(json.dumps(config,indent=2)+'\n')
    print('READY',family,flush=True)
