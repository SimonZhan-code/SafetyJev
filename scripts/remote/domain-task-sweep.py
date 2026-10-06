"""Run one complete benchmark level in the isolated ID/OOD workspace.

One pinned family policy at a time; full upstream episode caps, seed 0.
Clutter records simulator truth without any SafetyJev requests.
"""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
PY='/workspace/conda/behavior51/bin/python'
ENV=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_HEADLESS='1',
         OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets',
         CUDA_VISIBLE_DEVICES='0',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
         PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT)+':/workspace/ManiGuard')


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def service(action, name):
    return subprocess.run(['supervisorctl',action,name],capture_output=True,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--level',choices=['base','target','language','location','env'],required=True)
    parser.add_argument('--families',nargs='+',default=['jar','lid','stack','dusty','cabinet','clutter'])
    parser.add_argument('--max-scenes',type=int)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lock=Path('/workspace/SafetyJev/artifacts/base-sweep.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources=json.loads((ROOT/'artifacts/domain-sweep-resources.json').read_text())
    selected={f:{**resources['families'][f], 'scenes':resources['families'][f]['scenes_by_level'][args.level]} for f in args.families}
    if not all(s.get('ready') for s in selected.values()):raise RuntimeError('Downloads incomplete')
    plan={'families':selected,'seed':0,'max_scenes_override':args.max_scenes,
          'max_steps_override':args.max_steps,'sample_stride':8,'threshold':.5,
          'scope':args.level+' current predicate classification; no interventions', 'benchmark_level':args.level,
          'clutter':'VLA and simulator oracle only, requested by user',
          'checkpoint_selection':resources['checkpoint_selection'],
          'benchmark_revision':resources['benchmark_revision']}
    if (out/'plan.json').exists():
        if json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Resume plan differs')
    else:save(out/'plan.json',plan)
    source_paths=[*sorted((ROOT/'safetyjev').glob('*.py')), *sorted((ROOT/'configs').glob('*visual*.json')),
                  ROOT/'configs/domain-sweep-policies.json', ROOT/'scripts/remote/reset-domain-policy.py',
                  ROOT/'scripts/remote/domain-task-sweep.py']
    def source_hashes():
        return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    hashes=source_hashes()
    if (out/'source-sha256.json').exists():
        if json.loads((out/'source-sha256.json').read_text())!=hashes:raise ValueError('Runtime source changed since run started')
    else:save(out/'source-sha256.json',hashes)
    records=[]
    stop=threading.Event()
    def sample_gpu():
        with (out/'gpu.csv').open('a') as stream:
            if stream.tell()==0:stream.write('unix_s,memory_used_mib,utilization_percent\n')
            while not stop.is_set():
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
                stream.write(str(time.time())+','+result.stdout);stream.flush();stop.wait(1)
    worker=threading.Thread(target=sample_gpu,daemon=True);worker.start()
    try:
        for family,spec in selected.items():
            cases=spec['scenes'][:args.max_scenes] if args.max_scenes else spec['scenes']
            pending=[]
            for scene in cases:
                path=out/family/'cases'/(scene.replace('/','-')+'.json')
                if path.exists():
                    old=json.loads(path.read_text())
                    if old['status']=='completed' or (old['status']=='failed' and not args.retry_failed):
                        records.append(old);continue
                pending.append((scene,path))
            if not pending:continue
            if family!='clutter':
                result=service('start','safetyjev-domain-visual')
                deadline=time.monotonic()+240
                while True:
                    try:
                        with urlopen('http://127.0.0.1:8793/health',timeout=5) as response:health=json.load(response)
                        save(out/'classifier-health.json',health);break
                    except Exception:
                        if time.monotonic()>deadline:raise RuntimeError('Visual service unavailable: '+result.stdout)
                        time.sleep(2)
            else:service('stop','safetyjev-domain-visual')
            service('stop','safetyjev-domain-policy')
            save(ROOT/'artifacts/active-domain-policy.json',{'family':family})
            started=service('start','safetyjev-domain-policy')
            if started.returncode:raise RuntimeError(started.stdout+started.stderr)
            provenance=json.loads((ROOT/'configs/jar-isaac51-provenance.json').read_text())
            provenance.update(experiment='maniguard_id_ood_runtime_sweep',split=args.level,
                policy_repo=spec['repo'],policy_revision=spec['revision'],policy_checkpoint_subdirectory=spec['step'],
                policy_checkpoint_selection=resources['checkpoint_selection'],benchmark_revision=resources['benchmark_revision'])
            save(out/family/'provenance.json',provenance)
            for scene,path in pending:
                if source_hashes()!=hashes:raise ValueError('Runtime source changed during sweep')
                attempt=1
                if path.exists():attempt=json.loads(path.read_text()).get('attempt',1)+1
                rec={'family':family,'level':args.level,'scene':scene,'attempt':attempt,'status':'running','start_unix_s':time.time()}
                if path.exists():save(path.with_name(path.stem+f'.previous-{attempt-1}.json'),json.loads(path.read_text()))
                save(path,rec);save(out/'progress.json',{'status':'running','current':rec,'finished':len(records),'planned':sum(len(s['scenes'][:args.max_scenes] if args.max_scenes else s['scenes']) for s in selected.values())})
                print('START',family,scene,'attempt',attempt,flush=True)
                episodes=out/family/'episodes';previous=set(episodes.glob('*/episode.json'))
                log=out/family/'logs'/(scene.replace('/','-')+f'-{attempt}.log');log.parent.mkdir(parents=True,exist_ok=True)
                try:
                    warm=subprocess.run([PY,str(ROOT/'scripts/remote/reset-domain-policy.py'),'--family',family],env=ENV,capture_output=True,text=True,timeout=240)
                    if warm.returncode:raise RuntimeError('Policy warmup failed: '+warm.stderr[-1500:])
                    (out/family/'policy-health.json').write_text(warm.stdout)
                    command=[PY,'-u','-m','safetyjev.visual_runtime','capture','--maniguard-root','/workspace/ManiGuard',
                         '--output',str(episodes),'--provenance',str(out/family/'provenance.json')]
                    command+=['--oracle-only'] if family=='clutter' else ['--classifier-config',str(ROOT/f'configs/{family}-domain-visual.json')]
                    command+=['--','--config','configs/eval/'+spec['eval_config'],'--benchmark-root','/workspace/data/maniguard-bench/'+spec['pipeline'],
                         '--scenes',scene,'--seed','0','--port','8001','--max-steps',str(args.max_steps or spec['max_steps']),'--tag','safetyjev-domain-'+args.level]
                    with log.open('w') as stream:
                        subprocess.run(command,cwd='/workspace/ManiGuard',env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
                    added=set(episodes.glob('*/episode.json'))-previous
                    if len(added)!=1:raise RuntimeError('Expected one captured episode')
                    ep=next(iter(added)).parent;rec['episode_id']=ep.name
                    complete=json.loads((ep/'complete.json').read_text());result=json.loads((ep/'maniguard_result.json').read_text())
                    if complete['status']!='completed' or not complete['monitor_valid']:raise RuntimeError('Invalid completed episode')
                    if family!='clutter' and result['safetyjev_classification']['failed_samples']:raise RuntimeError('Classifier failures')
                    rec.update(status='completed',result=result)
                except Exception as exc:
                    rec.update(status='failed',error=type(exc).__name__+': '+str(exc))
                    rec['captured_episode_ids']=[p.parent.name for p in set(episodes.glob('*/episode.json'))-previous]
                rec.update(end_unix_s=time.time(),log=str(log.relative_to(out)))
                save(path,rec);records.append(rec);save(out/'sweep.json',records)
                subprocess.run([PY,str(ROOT/'scripts/remote/summarize-base-sweep.py'),str(out)],check=True)
                print('DONE',family,scene,rec['status'],rec.get('error',''),flush=True)
            if family!='clutter':
                subprocess.run([PY,'-m','safetyjev.visual_runtime','report','--episodes',str(out/family/'episodes'),'--output',str(out/family/'report.json')],env=ENV,cwd=ROOT,check=True)
                audited=subprocess.run([PY,str(ROOT/'scripts/remote/audit-visual-evaluation.py'),str(out/family),
                    '--calibration','/workspace/checkpoints/safetyjev/round2-amd-20k/calibration.json',
                    '--output',str(out/family/'audit-summary.json')],env=ENV,cwd=ROOT,capture_output=True,text=True)
                save(out/family/'audit-status.json',{'passed':audited.returncode==0,'stdout':audited.stdout,'stderr':audited.stderr})
            subprocess.run([PY,str(ROOT/'scripts/remote/summarize-base-sweep.py'),str(out)],check=True)
            service('stop','safetyjev-domain-policy')
        save(out/'progress.json',{'status':'finished','completed':sum(r['status']=='completed' for r in records),'failed':sum(r['status']=='failed' for r in records),'planned':len(records)})
    finally:
        service('stop','safetyjev-domain-policy');service('stop','safetyjev-domain-visual')
        stop.set();worker.join(timeout=5)
    if any(r['status']!='completed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
