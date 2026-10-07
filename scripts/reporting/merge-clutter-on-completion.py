"""One-shot Clutter backup, merge, and completion handoff; no chat automation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/clutter-worker-20261006'
SOURCE=['ssh','-p','58371','-o','BatchMode=yes','-o','ConnectTimeout=20','root@47.186.21.5']
DEST=['ssh','-p','39237','-o','BatchMode=yes','-o','ConnectTimeout=20','root@154.59.156.14']
RUN='/workspace/SafetyJev/artifacts/clutter-base-20261006'
BASE='/workspace/SafetyJev/artifacts/base-sweep-20261006'
ARCHIVE='/workspace/SafetyJev/artifacts/clutter-complete-20261006.tar.gz'


def save(value):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'merge-status.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(OUT/'merge-status.json')


def query(ssh,script):
    result=subprocess.run(ssh+['python3 -'],input=script,text=True,capture_output=True,timeout=120)
    if result.returncode:raise RuntimeError('Remote state check failed')
    return json.loads(result.stdout)


def terminal_55(progress,cases):
    return (progress.get('status')=='finished' and progress.get('planned')==55
            and len(cases)==55 and {c.get('scene') for c in cases}=={f'task_{i:04d}/base' for i in range(55)}
            and all(c.get('family')=='clutter' and c.get('status') in ('completed','failed') for c in cases))


def main():
    save({'status':'waiting_for_55_clutter_cases'})
    script=f'''import json
from pathlib import Path
r=Path({RUN!r})
p=json.loads((r/'progress.json').read_text()) if (r/'progress.json').exists() else {{}}
c=[json.loads(x.read_text()) for x in (r/'clutter/cases').glob('task_*-base.json')]
print(json.dumps({{'progress':p,'cases':c}}))
'''
    while True:
        try:data=query(SOURCE,script)
        except (RuntimeError,subprocess.TimeoutExpired):time.sleep(30);continue
        if terminal_55(data['progress'],data['cases']):break
        time.sleep(20)
    save({'status':'archiving_clutter'})
    script=f'''import json,tarfile,hashlib
from pathlib import Path
r=Path({RUN!r});a=Path({ARCHIVE!r});tmp=a.with_suffix('.partial')
with tarfile.open(tmp,'w:gz',compresslevel=1) as t:
 t.add(r/'clutter',arcname='clutter')
 for name in ['plan.json','source-sha256.json','progress.json','gpu.csv']:
  t.add(r/name,arcname='clutter-worker-'+name)
with tmp.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
tmp.replace(a)
print(json.dumps({{'sha256':h,'bytes':a.stat().st_size,'archive':str(a)}}))
'''
    # Compression can take longer than a state check.
    result=subprocess.run(SOURCE+['python3 -'],input=script,text=True,capture_output=True,timeout=1800)
    if result.returncode:raise RuntimeError('Clutter archive failed')
    meta=json.loads(result.stdout);archive=OUT/'clutter-complete-20261006.tar.gz'
    save({'status':'downloading',**meta})
    subprocess.run(['rsync','-a','--partial','-e','ssh -p 58371 -o BatchMode=yes -o ConnectTimeout=20',
                    'root@47.186.21.5:'+ARCHIVE,str(archive)],check=True)
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=meta['sha256'] or archive.stat().st_size!=meta['bytes']:raise RuntimeError('Clutter backup verification failed')
    with tarfile.open(archive) as t:t.extractall(OUT/'results',filter='data')
    (OUT/'archive-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save({'status':'verified_local_backup_waiting_for_cabinet',**meta})
    check=f'''import json,subprocess
from pathlib import Path
r=Path({BASE!r});c=[json.loads(p.read_text()) for p in (r/'cabinet/cases').glob('task_*-base.json')]
state=subprocess.run(['supervisorctl','status','safetyjev-base-sweep'],capture_output=True,text=True).stdout
print(json.dumps({{'cabinet_terminal':len(c)==35 and all(x['status'] in ('completed','failed') for x in c),'runner_active':'RUNNING' in state,'clutter_cases':len(list((r/'clutter/cases').glob('task_*-base.json')))}}))
'''
    while True:
        state=query(DEST,check)
        if state['cabinet_terminal'] and not state['runner_active']:
            if state['clutter_cases']:raise RuntimeError('Destination already has Clutter cases; refuse overwrite')
            break
        time.sleep(20)
    remote='/workspace/SafetyJev/artifacts/clutter-import.tar.gz'
    subprocess.run(['rsync','-a','--partial','-e','ssh -p 39237 -o BatchMode=yes -o ConnectTimeout=20',str(archive),'root@154.59.156.14:'+remote],check=True)
    script=f'''import hashlib,json,tarfile,subprocess
from pathlib import Path
a=Path({remote!r});r=Path({BASE!r})
with a.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={digest!r}
assert not list((r/'clutter/cases').glob('task_*-base.json'))
with tarfile.open(a) as t:t.extractall(r,filter='data')
receipt={{'status':'verified_clutter_import','source_instance_id':54566989,'sha256':{digest!r},'cases':55}}
(r/'clutter-import-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
subprocess.run(['supervisorctl','start','safetyjev-base-sweep'],check=True,stdout=subprocess.DEVNULL)
print(json.dumps(receipt))
'''
    receipt=query(DEST,script);save({**receipt,'status':'clutter_imported_waiting_for_200_terminal'})
    while True:
        state=query(DEST,f'''import json
from pathlib import Path
r=Path({BASE!r});p=json.loads((r/'progress.json').read_text())
j=json.loads(Path('/workspace/SafetyJev-ood-20261006/artifacts/domain-evaluation-20261006/jar-base/progress.json').read_text())
print(json.dumps({{'base':p,'jar':j}}))
''')
        a,b=state['base'],state['jar']
        if (a.get('status')=='finished' and a.get('planned')==174 and a.get('completed',0)+a.get('failed',0)==174
                and b.get('status')=='finished' and b.get('planned')==26 and b.get('completed',0)+b.get('failed',0)==26):break
        time.sleep(5)
    cli='/Users/simonzhan/miniconda3/bin/vastai'
    result=subprocess.run([cli,'stop','instance','54566989'],capture_output=True,text=True,timeout=120)
    accepted=result.returncode==0 and 'stopping instance 54566989.' in result.stdout
    save({'status':'completed_and_new_node_stop_requested' if accepted else 'completed_but_new_node_stop_failed','sha256':digest,'source_instance_id':54566989,'stop_accepted':accepted})
    if not accepted:raise RuntimeError('Vast stop request was not accepted')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise
