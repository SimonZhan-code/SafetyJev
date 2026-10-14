"""Read-only sweep health probes; --watch writes only a separate health snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time


def snapshot(root, job):
    alerts=[]
    result={'time':time.time(),'root':str(root),'alerts':alerts}
    for name in ('progress','summary'):
        p=root/(name+'.json')
        if p.exists():result[name]={k:v for k,v in json.loads(p.read_text()).items() if k!='records'}
    p=root/'plan.json'
    if p.exists():
        plan=json.loads(p.read_text())
        result['hash_mismatches']=[f for f,h in plan['source_sha256'].items()
            if not Path(f).exists() or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
        if result['hash_mismatches']:alerts.append('runtime_hash_mismatch')
    state=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=15).stdout
    names=(job,'safetyjev-codex-predictor','safetyjev-codex-pi05')
    result['services']=[l for l in state.splitlines() if l.split() and l.split()[0] in names]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    result['gpu']=gpu.stdout.strip();result['free_gib']=round(shutil.disk_usage(root).free/1024**3,2)
    if gpu.returncode:alerts.append('gpu_probe_failed')
    if result['free_gib']<20:alerts.append('disk_reserve_low')
    progress=result.get('progress',{});running=progress.get('status')=='running'
    if progress.get('status') in ('failed','blocked_preflight'):alerts.append('queue_failed')
    if running and not any(l.startswith(job+' ') and 'RUNNING' in l for l in result['services']):alerts.append('queue_not_running')
    current=progress.get('current')
    if current:
        case=root/(current['scene'].replace('/','-')+'-seed'+str(current['seed']))
        episodes=list((case/'episodes').glob('*/harness-events.jsonl'))
        row={'scene':current['scene'],'seed':current['seed'],'steps':0}
        if episodes:
            f=max(episodes,key=lambda x:x.stat().st_mtime);events=[]
            for line in f.read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass  # In-progress final line.
            row['event_age_s']=round(time.time()-f.stat().st_mtime,1)
            row['skills_started']=sum(e['event']=='skill_start' for e in events)
            row['chunks_committed']=sum(e['event']=='selection' for e in events)
            row['failures']=[e.get('error') for e in events if e['event']=='failure']
            row['tool_error_count']=sum(e['event']=='tool' and 'error' in e.get('response',{}) for e in events)
            p=f.parent/'executed-candidates.jsonl'
            if p.exists():row['steps']=sum(1 for _ in p.open())
            if running and row['event_age_s']>300:alerts.append('no_harness_event_5min')
            if row['failures'] or row['tool_error_count']:alerts.append('harness_or_tool_error')
        else:
            files=[p for p in case.glob('*') if p.is_file()]
            row['startup_activity_age_s']=round(time.time()-max((p.stat().st_mtime for p in files),default=time.time()),1)
            if running and row['startup_activity_age_s']>600:alerts.append('startup_stall_10min')
        result['current']=row
    result['service_tcp']={}
    for name,port in (('safetyjev-codex-predictor',8794),('safetyjev-codex-pi05',8000)):
        if any(l.startswith(name+' ') and 'RUNNING' in l for l in result['services']):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=3):pass
                result['service_tcp'][name]=True
            except OSError:result['service_tcp'][name]=False  # Startup is allowed; heartbeat correlates age.
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',default='safetyjev-codex-jar-eval-skills');p.add_argument('--watch',type=int,default=0);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.watch and (a.watch<30 or not a.output):p.error('watch requires >=30 seconds and output')
    while True:
        try:r=snapshot(a.root,a.job)
        except Exception as exc:r={'time':time.time(),'alerts':['monitor_probe_failed'],'error_type':type(exc).__name__}
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp')
            tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output)
        if not a.watch:print(json.dumps(r,indent=2));return
        time.sleep(a.watch)


if __name__=='__main__':main()
