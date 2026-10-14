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
    if suite == 'single-smoke':
        return [('task_0000/base', 1)]
    if suite == 'timing':
        return [('task_0000/base', 1)]
    if suite == 'jar-base':
        return [(f'task_{i:04d}/base', seed) for i in range(26) for seed in range(3)]
    raise ValueError('Unknown Codex suite')


SINGLE_SMOKE_LAUNCHER_SHA256 = '68f9a9340a6e78c37db90f93f30b160f077d7fd2e05aaec143ccec11ebaf1ede'


def verify_smoke(directory, config, source_hashes, config_path, authorized_single=False):
    """A longer budget cannot bypass live mechanics and source validation."""
    if directory is None:
        raise ValueError('Jar sweep requires --smoke-evidence')
    plan=json.loads((directory/'plan.json').read_text())
    progress=json.loads((directory/'progress.json').read_text())
    count=1 if authorized_single else 2
    if authorized_single and config.get('tool_sequence')!='skills':
        raise ValueError('Single-smoke authorization applies only to bounded skills')
    if progress != {'status':'completed','completed':count,'planned':count}:
        raise ValueError('Both smoke trials must complete' if not authorized_single else 'Single smoke must complete')
    expected=dict(config,max_steps=256)
    suite='single-smoke' if authorized_single else 'smoke'
    if plan.get('suite')!=suite or plan.get('cases') != [[s,n] for s,n in case_grid(suite)]:
        raise ValueError('Unexpected smoke case grid')
    if plan['config'] != expected or plan['maniguard'] != jar.source_provenance():
        raise ValueError('Smoke model/controller or ManiGuard identity changed')
    for path, sha in source_hashes.items():
        # Config is compared structurally above; its action cap intentionally differs.
        if authorized_single and path == str(Path(__file__).resolve()) and plan['source_sha256'].get(path)==SINGLE_SMOKE_LAUNCHER_SHA256:
            continue  # Explicitly pinned prior launcher; all runtime/model/scene hashes still match.
        if path != str(config_path.resolve()) and plan['source_sha256'].get(path) != sha:
            raise ValueError('Smoke runtime differs from sweep')
    for scene,seed in case_grid(suite):
        case=directory/(scene.replace('/','-')+('-seed'+str(seed) if authorized_single else ''))
        episodes=list((case/'episodes').glob('*/complete.json'))
        if len(episodes)!=1 or audit(episodes[0].parent)['status']!='completed':
            raise ValueError('Smoke execution audit failed')


def completed_predecessor(directory):
    """Exclude audited terminal cases while retaining their distinct experiment identity."""
    if directory is None:
        return []
    directory=directory.resolve()
    plan=json.loads((directory/'plan.json').read_text())
    if (plan.get('suite')!='jar-base' or plan.get('max_steps')!=2000 or
            plan.get('maniguard')!=jar.source_provenance()):
        raise ValueError('Incompatible predecessor suite')
    rows=[]
    for scene,seed in case_grid('jar-base'):
        case=directory/(scene.replace('/','-')+'-seed'+str(seed))
        record=case/'record.json'
        if not record.exists() or json.loads(record.read_text()).get('status')!='completed':
            continue
        eps=list((case/'episodes').glob('*/complete.json'))
        if len(eps)!=1 or audit(eps[0].parent)['status']!='completed':
            raise ValueError('Predecessor execution audit failed')
        rows.append({'scene':scene,'seed':seed,'record':str(record),
            'record_sha256':hashlib.sha256(record.read_bytes()).hexdigest(),
            'plan_sha256':hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()})
    return rows


TIMEOUT_REASONS = frozenset(('decision_timeout', 'skill_or_planner_timeout'))


def terminal_report(case, scene, seed, wall_seconds, returncode, continue_on_timeout):
    """Only a fully audited, explicit planner/skill timeout may advance the queue.

    Missing/corrupt captures, predictor errors, audit failures and unexplained nonzero
    exits remain fatal. Partial actions and raw outcomes stay in the original report.
    """
    eps=list((case/'episodes').glob('*/complete.json'))
    if len(eps)!=1:
        raise RuntimeError('Expected one terminal episode')
    report=audit(eps[0].parent)
    report.update(scene=scene,seed=seed,wall_seconds=wall_seconds,capture_exit_code=returncode)
    reason=report.get('result',{}).get('safetyjev_codex',{}).get('infrastructure_failure')
    if continue_on_timeout and report['status']=='infrastructure_failure' and reason in TIMEOUT_REASONS:
        report.update(status='timeout',failure_reason=reason,queue_action='continue_next_case',
                      execution_audit_passed=True)
    elif returncode and report['status']=='completed':
        raise RuntimeError('Capture exited unsuccessfully despite completed report')
    return report


def skip_terminal_case(case, continue_on_timeout):
    """A restart must not rerun either completed cases or recorded timeout attempts."""
    record=case/'record.json'
    if not record.exists():return False
    old=json.loads(record.read_text())
    if old['status']=='completed':
        eps=list((case/'episodes').glob('*/complete.json'))
        if len(eps)!=1 or audit(eps[0].parent)['status']!='completed':
            raise RuntimeError('Completed case failed re-audit')
        return True
    if old['status']=='timeout':
        if not continue_on_timeout:raise RuntimeError('Timeout record requires its original queue policy')
        checked=terminal_report(case,old['scene'],old['seed'],old['wall_seconds'],old['capture_exit_code'],True)
        if (checked['status']!='timeout' or checked['failure_reason']!=old['failure_reason'] or
                checked['executed_steps']!=old['executed_steps']):
            raise RuntimeError('Timeout case failed re-audit')
        return True
    return False


def final_progress(records, planned):
    summary=summarize(records,planned)
    if summary['completed']+summary['timeouts']!=planned:
        raise RuntimeError('Queue has unprocessed or fatal cases')
    if not summary['timeouts']:
        return {'status':'completed','completed':planned,'planned':planned}
    return {'status':'completed_with_timeouts','planned':planned,'completed':summary['completed'],
            'timeouts':summary['timeouts'],'attempted':planned,'failed':summary['failed']}


def summarize(records, planned):
    done=[r for r in records if r.get('status')=='completed']
    return {'planned':planned,'completed':len(done),
        'failed':sum(r.get('status') in ('failed','infrastructure_failure','timeout') for r in records),
        'timeouts':sum(r.get('status')=='timeout' for r in records),
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
        'proposal_sources':{s:sum(r['proposal_sources'].get(s,0) for r in done) for s in sorted({'pi05','codex'} | {s for r in done for s in r['proposal_sources']})},
        'predictor_blocked':sum((r.get('skills') or {}).get('policy_stop')=='predictor_blocked' for r in done),
        'planner_budget_stops':sum((r.get('skills') or {}).get('policy_stop')=='planner_call_budget' for r in done),
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
    p.add_argument('--suite',choices=('smoke','single-smoke','jar-base','timing'),default='smoke')
    p.add_argument('--smoke-evidence',type=Path)
    p.add_argument('--authorized-single-smoke',action='store_true',help='Explicit user approval to admit bounded skills from the audited single smoke')
    p.add_argument('--exclude-completed-from',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/codex-harness.example.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/codex-harness-smoke')
    p.add_argument('--check',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--continue-on-timeout',action='store_true',help='Record audited planner/skill timeouts and advance without retrying them')
    args=p.parse_args();os.umask(0o077)
    config=load_config(args.config)
    if args.continue_on_timeout and args.suite!='jar-base':
        raise ValueError('Timeout continuation is for full sweeps only; smoke timeouts block admission')
    if args.authorized_single_smoke and args.suite!='jar-base':
        raise ValueError('Single-smoke admission is for full sweep only')
    cap=256 if args.suite in ('smoke','single-smoke') else 2000
    if config['max_steps']!=cap:raise ValueError('Incorrect action cap for selected suite')
    if args.exclude_completed_from and args.suite!='jar-base':
        raise ValueError('Only full sweeps may exclude predecessor cases')
    excluded=completed_predecessor(args.exclude_completed_from)
    excluded_keys={(r['scene'],r['seed']) for r in excluded}
    grid=[c for c in case_grid(args.suite) if c not in excluded_keys]
    if not grid:raise ValueError('No remaining cases')
    JOB={'smoke':'safetyjev-codex-smoke', 'jar-base':'safetyjev-codex-jar-eval',
         'single-smoke':'safetyjev-codex-single-smoke',
         'timing':'safetyjev-codex-timing'}[args.suite]
    if config['tool_sequence']=='skills':JOB += '-skills'
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
           ROOT/'scripts/remote/jar-agentic-eval.py',ROOT/'scripts/remote/serve-pi05.sh',
           ROOT/'scripts/remote/warm-jar-policy.py',Path(jar.MANIGUARD)/'configs/eval/jar_transport_joint.yaml',
           ROOT/'configs/jar-isaac51-provenance.json',ROOT/'configs/predictor-semantic-definitions.json']
    for scene,_ in case_grid('jar-base')[::3]:
        if not list((jar.BENCH/scene).glob('scene*.json')):
            raise RuntimeError('Missing pinned Jar scene')
        paths.extend(sorted((jar.BENCH/scene).glob('*.json*')))
    plan={'maniguard':jar.source_provenance(),'suite':args.suite,'cases':[[s,n] for s,n in grid],'max_steps':cap,'mode':'codex_harness','config':config,
          'source_sha256':{str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
    if excluded:plan['excluded_predecessor_cases']=excluded
    if args.continue_on_timeout:plan['timeout_policy']='record_and_continue'
    if args.suite=='jar-base':
        verify_smoke(args.smoke_evidence,config,plan['source_sha256'],args.config,args.authorized_single_smoke)
        plan['smoke_admission']={'authorized_single':args.authorized_single_smoke,'evidence':str(args.smoke_evidence.resolve())}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise RuntimeError('Plan changed; use a new output directory')
    write_json(out/'plan.json',plan)
    if args.check:return
    no_overlap()
    if args.launch:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--suite',args.suite,'--config',str(args.config.resolve()),'--output',str(out)]
        if args.authorized_single_smoke:command.append('--authorized-single-smoke')
        if args.smoke_evidence:command+=['--smoke-evidence',str(args.smoke_evidence.resolve())]
        if args.exclude_completed_from:command+=['--exclude-completed-from',str(args.exclude_completed_from.resolve())]
        if args.retry_failed:command.append('--retry-failed')
        if args.continue_on_timeout:command.append('--continue-on-timeout')
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
            if skip_terminal_case(case,args.continue_on_timeout):continue
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
                    capture=subprocess.run(command,cwd=jar.MANIGUARD,env=env,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=cap//8*config['decision_timeout_s']+900)
                report=terminal_report(case,scene,seed,time.monotonic()-started,capture.returncode,args.continue_on_timeout)
                write_json(record,report)
                if report['status'] not in ('completed','timeout'):raise RuntimeError('Harness infrastructure failure')
                if report['status']=='timeout':
                    # The episode watchdog closes the worker and returns a terminal capture.
                    # Stop inference before advancing; every next case starts fresh services.
                    for name in reversed(SERVICES):
                        if name in active:
                            jar.service('stop',name);active.remove(name)
            except Exception:
                if json.loads(record.read_text())['status']=='running':write_json(record,{'status':'failed','scene':scene,'seed':seed})
                write_json(out/'progress.json',{'status':'failed','current':{'scene':scene,'seed':seed},**summarize(records(),len(grid))})
                raise
            write_json(out/'summary.json',summarize(records(),len(grid)))
        final=final_progress(records(),len(grid))
        write_json(out/'summary.json',{**summarize(records(),len(grid)),**final,'records':records()})
        write_json(out/'progress.json',final)
    finally:
        for name in reversed(SERVICES):
            if name in active:jar.service('stop',name)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('Codex smoke did not complete: '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1) from None
