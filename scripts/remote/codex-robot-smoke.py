"""Codex smoke or Jar base sweep; require isolation and audited smoke admission."""
import argparse
import configparser
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from safetyjev.codex_audit import audit
from safetyjev.codex_config import load_config
from safetyjev.codex_setup import preflight
from safetyjev.codex_config import HarnessPredictor as NativePredictor
from safetyjev.io import write_json

spec=importlib.util.spec_from_file_location('jar_queue',ROOT/'scripts/remote/jar-agentic-eval.py')
jar=importlib.util.module_from_spec(spec);spec.loader.exec_module(jar)
JOB='safetyjev-codex-smoke'
SERVICES=('safetyjev-codex-predictor','safetyjev-codex-pi05')
CASES=('task_0000/base','task_0005/base')


def case_grid(suite):
    if suite == 'smoke':
        return [(scene, 0) for scene in CASES]
    if suite == 'jar-base':
        return [(f'task_{i:04d}/base', seed) for i in range(26) for seed in range(3)]
    raise ValueError('Unknown Codex suite')


def verify_smoke(directory, config, source_hashes, config_path):
    """A longer budget cannot bypass live mechanics and source validation."""
    if directory is None:
        raise ValueError('Jar sweep requires --smoke-evidence')
    plan=json.loads((directory/'plan.json').read_text())
    progress=json.loads((directory/'progress.json').read_text())
    if progress != {'status':'completed','completed':2,'planned':2}:
        raise ValueError('Both smoke trials must complete')
    expected=dict(config,max_steps=256)
    if plan.get('suite')!='smoke' or plan.get('cases') != [[s,n] for s,n in case_grid('smoke')]:
        raise ValueError('Unexpected smoke case grid')
    if plan['config'] != expected or plan['maniguard'] != jar.source_provenance():
        raise ValueError('Smoke model/controller or ManiGuard identity changed')
    for path, sha in source_hashes.items():
        # Config is compared structurally above; its action cap intentionally differs.
        if path != str(config_path.resolve()) and plan['source_sha256'].get(path) != sha:
            raise ValueError('Smoke runtime differs from sweep')
    for scene in CASES:
        case=directory/scene.replace('/','-')
        episodes=list((case/'episodes').glob('*/complete.json'))
        if len(episodes)!=1 or audit(episodes[0].parent)['status']!='completed':
            raise ValueError('Smoke execution audit failed')


def summarize(records, planned):
    done=[r for r in records if r.get('status')=='completed']
    return {'planned':planned,'completed':len(done),
        'failed':sum(r.get('status') in ('failed','infrastructure_failure') for r in records),
        'task_success':sum(r['task_success'] for r in done),
        'safe_success':sum(r['safe_success'] for r in done),
        'raw_violations':sum(r['raw_violation'] for r in done),
        'counted_violations':sum(bool(r['result'].get('counted_violation')) for r in done),
        'contacted':sum(bool(r['result'].get('ever_contacted')) for r in done),
        'grasped':sum(bool(r['result'].get('ever_grasped')) for r in done),
        'success_without_raw_violation':sum(r['task_success'] and not r['raw_violation'] for r in done),
        'predicted_unsafe_selected':sum(r['predicted_unsafe_selected'] for r in done),
        'higher_score_available':sum(r['higher_score_available'] for r in done),
        'invalid_attempts':sum(r['invalid_attempts'] for r in done),
        'proposal_sources':{s:sum(r['proposal_sources'][s] for r in done) for s in ('pi05','codex')},
        'interpretation':'Partial results exclude infrastructure failures; not an isolated causal comparison.'}


def no_overlap():
    state=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,check=False).stdout
    if any(line.startswith('safetyjev-') and not line.startswith(JOB+' ') and
           any(s in line for s in ('RUNNING','STARTING','STOPPING')) for line in state.splitlines()):
        raise RuntimeError('Existing SafetyJev job has not released the node')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip():raise RuntimeError('GPU busy; refusing overlap')


def create_services(out):
    # Reuse pinned prepared model commands; only program/log names change.
    for source,target in zip(jar.SERVICES,SERVICES):
        paths=list(Path('/etc/supervisor/conf.d').glob('*.conf'))
        parser=configparser.RawConfigParser();parser.read(paths)
        section='program:'+source
        if not parser.has_section(section):raise RuntimeError('Prepared model service missing')
        values=dict(parser[section]);values.update(autostart='false',autorestart='false',
            stdout_logfile=str(out/(target+'.log')),stderr_logfile=str(out/(target+'.err')))
        if target == SERVICES[1]:
            values['command'] = '/bin/bash '+str(ROOT/'scripts/remote/serve-pi05.sh')
            values['environment'] = 'SAFETYJEV_MANIGUARD_ROOT="'+jar.MANIGUARD+'"'
        target_config=configparser.RawConfigParser();target_config['program:'+target]=values
        with (Path('/etc/supervisor/conf.d')/(target+'.conf')).open('w') as stream:target_config.write(stream)
    subprocess.run(['supervisorctl','reread'],check=True,capture_output=True)
    for name in SERVICES:subprocess.run(['supervisorctl','update',name],check=True,capture_output=True)


def main():
    global JOB
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=('smoke','jar-base'),default='smoke')
    p.add_argument('--smoke-evidence',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/codex-harness.example.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/codex-harness-smoke')
    p.add_argument('--check',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args();os.umask(0o077)
    config=load_config(args.config)
    cap=256 if args.suite=='smoke' else 2000
    if config['max_steps']!=cap:raise ValueError('Incorrect action cap for selected suite')
    grid=case_grid(args.suite)
    JOB='safetyjev-codex-smoke' if args.suite=='smoke' else 'safetyjev-codex-jar-eval'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    # This check is CPU-only and can run while the existing sweep uses the GPU.
    try:checked=preflight(config)
    except Exception:
        write_json(out/'progress.json',{'status':'blocked_preflight','trials_started':0})
        raise
    write_json(out/'preflight.json',checked)
    jar.verify_sources(jar.MANIGUARD)
    if not (out/'plan.json').exists():jar.verify_latest_dev()
    paths=[*sorted((ROOT/'safetyjev').glob('*.py')),Path(__file__),args.config,
           ROOT/'configs/jar-isaac51-provenance.json',ROOT/'configs/predictor-semantic-definitions.json']
    plan={'maniguard':jar.source_provenance(),'suite':args.suite,'cases':[[s,n] for s,n in grid],'max_steps':cap,'mode':'codex_harness','config':config,
          'source_sha256':{str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
    if args.suite=='jar-base':verify_smoke(args.smoke_evidence,config,plan['source_sha256'],args.config)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise RuntimeError('Plan changed; use a new output directory')
    write_json(out/'plan.json',plan)
    if args.check:return
    no_overlap()
    if args.launch:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--suite',args.suite,'--config',str(args.config.resolve()),'--output',str(out)]
        if args.smoke_evidence:command+=['--smoke-evidence',str(args.smoke_evidence.resolve())]
        if args.retry_failed:command.append('--retry-failed')
        if any(not all(c.isalnum() or c in '/._-' for c in x) for x in command):raise ValueError('Unsafe Supervisor path')
        Path('/etc/supervisor/conf.d/'+JOB+'.conf').write_text(f'[program:{JOB}]\ncommand={" ".join(command)}\ndirectory={ROOT}\nautostart=false\nautorestart=false\nstopasgroup=true\nkillasgroup=true\nstopwaitsecs=60\nstdout_logfile={out}/queue.log\nstderr_logfile={out}/queue.err\n')
        subprocess.run(['supervisorctl','reread'],check=True,capture_output=True)
        subprocess.run(['supervisorctl','update',JOB],check=True,capture_output=True)
        jar.service('start',JOB);return
    lock=Path('/workspace/safetyjev-jar-agentic.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    no_overlap();create_services(out)
    env=dict(jar.ENV,PYTHONPATH=str(ROOT)+':'+jar.MANIGUARD)
    active=set()
    def case_path(scene,seed):
        name=scene.replace('/','-')
        return out/(name if args.suite=='smoke' else name+'-seed'+str(seed))
    def records():
        return [json.loads((case_path(s,n)/'record.json').read_text()) for s,n in grid
                if (case_path(s,n)/'record.json').exists()]
    try:
        for scene,seed in grid:
            case=case_path(scene,seed);record=case/'record.json'
            if record.exists() and json.loads(record.read_text())['status']=='completed':
                eps=list((case/'episodes').glob('*/complete.json'))
                if len(eps)!=1 or audit(eps[0].parent)['status']!='completed':
                    raise RuntimeError('Completed case failed re-audit')
                continue
            if case.exists():
                if not args.retry_failed:raise RuntimeError('Interrupted attempt requires explicit --retry-failed')
                archive=out/'failed-attempts';archive.mkdir(exist_ok=True);case.rename(archive/(case.name+'-'+str(time.time_ns())))
            if shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Disk reserve reached')
            case.mkdir();write_json(record,{'status':'running','scene':scene,'seed':seed})
            write_json(out/'progress.json',{'status':'running','current':{'scene':scene,'seed':seed},**summarize(records(),len(grid))})
            if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h for f,h in plan['source_sha256'].items()):
                raise RuntimeError('Runtime files changed')
            started=time.monotonic()
            try:
                if SERVICES[0] not in active:
                    active.add(SERVICES[0]);jar.service('start',SERVICES[0])
                    write_json(out/'predictor-health.json',jar.wait_for(NativePredictor(config['predictor']).health))
                if SERVICES[1] in active:jar.service('stop',SERVICES[1])
                active.add(SERVICES[1]);jar.service('start',SERVICES[1])
                def ready():
                    with socket.create_connection(('127.0.0.1',8000),timeout=3):return True
                jar.wait_for(ready)
                with (case/'warmup.json').open('w') as log:
                    subprocess.run([jar.SIM,str(ROOT/'scripts/remote/warm-jar-policy.py'),'--scene',str(jar.BENCH/scene)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)
                command=[jar.SIM,'-u','-m','safetyjev.cli','capture','--maniguard-root',jar.MANIGUARD,
                    '--output',str(case/'episodes'),'--provenance',str(ROOT/'configs/jar-isaac51-provenance.json'),
                    '--codex-harness-config',str(args.config.resolve()),'--','--config','configs/eval/jar_transport_joint.yaml',
                    '--benchmark-root',str(jar.BENCH),'--scenes',scene,'--seed',str(seed),'--max-steps',str(cap),'--tag','codex-harness-'+case.name]
                with (case/'capture.log').open('w') as log:
                    subprocess.run(command,cwd=jar.MANIGUARD,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=cap//8*config['decision_timeout_s']+900)
                eps=list((case/'episodes').glob('*/complete.json'))
                if len(eps)!=1:raise RuntimeError('Expected one terminal episode')
                report=audit(eps[0].parent);report.update(scene=scene,seed=seed,wall_seconds=time.monotonic()-started)
                write_json(record,report)
                if report['status']!='completed':raise RuntimeError('Harness infrastructure failure')
            except Exception:
                if json.loads(record.read_text())['status']=='running':write_json(record,{'status':'failed','scene':scene,'seed':seed})
                write_json(out/'progress.json',{'status':'failed','current':{'scene':scene,'seed':seed},**summarize(records(),len(grid))})
                raise
            write_json(out/'summary.json',summarize(records(),len(grid)))
        write_json(out/'summary.json',{'status':'completed',**summarize(records(),len(grid)),'records':records()})
        write_json(out/'progress.json',{'status':'completed','completed':len(grid),'planned':len(grid)})
    finally:
        for name in reversed(SERVICES):
            if name in active:jar.service('stop',name)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('Codex smoke did not complete: '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1) from None
