import os,json,time,shutil,hashlib
from pathlib import Path
src=Path('/workspace');dst=src/'clutter-runtime-transfer';dst.mkdir(exist_ok=True)
items=['conda','BEHAVIOR-5.1','ManiGuard','openpi','checkpoints/pi05-clutter','data/maniguard-bench/clutter_pickup']
skip={'.git','__pycache__','.env','token','stored_tokens','results','outputs'}
files=[]
for item in items:
 root=src/item
 for folder,dirs,names in os.walk(root,followlinks=False):
  rel=Path(folder).relative_to(src);target=dst/rel;target.mkdir(parents=True,exist_ok=True)
  dirs[:]=[d for d in dirs if d not in skip]
  for name in list(dirs)+names:
   if name in skip or name.endswith('.log'):continue
   a=Path(folder)/name;b=target/name
   if a.is_symlink():
    if not b.is_symlink():b.symlink_to(os.readlink(a))
   elif a.is_file():
    if not b.exists():os.link(a,b)
    files.append({'path':str(a.relative_to(src)),'bytes':a.stat().st_size})
 # Directory metadata matters for packaged environment paths.
 print('STAGED',item,flush=True)
manifest={'status':'staged','source_instance_id':54533396,'destination_instance_id':54566989,'files':files,'file_count':len(files),'bytes':sum(f['bytes'] for f in files),'unix_s':time.time()}
(dst/'runtime-transfer-manifest.json').write_text(json.dumps(manifest)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)
