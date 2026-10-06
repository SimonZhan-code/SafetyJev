import concurrent.futures,json,subprocess,time
from pathlib import Path
root=Path('/workspace/bootstrap-copy')
items=['conda','BEHAVIOR-5.1','ManiGuard','openpi','Open-Jev-visual-pinned','checkpoints','data','.hf_home/hub','SafetyJev','SafetyJev-ood-20261006']
def copy(name):
 dest=Path('/workspace')/name;dest.mkdir(parents=True,exist_ok=True)
 cmd=['rsync','-aH','--partial','--stats','--exclude=.git','--exclude=__pycache__','--exclude=*.log','--exclude=.env','--exclude=token','--exclude=stored_tokens']
 if name.startswith('SafetyJev'):cmd+=['--exclude=/artifacts/']
 cmd+=['-e','ssh -p 15019 -i /workspace/bootstrap-copy/id_ed25519 -o BatchMode=yes','root@87.192.101.6:'+name+'/',str(dest)+'/']
 print('START',name,flush=True)
 with (root/(name.replace('/','_')+'.log')).open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(name+' transfer exit '+str(result.returncode))
 print('DONE',name,flush=True)
 return name
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 copied=list(pool.map(copy,items))
(root/'complete.json').write_text(json.dumps({'status':'completed','copied':copied,'unix_s':time.time()},indent=2))
