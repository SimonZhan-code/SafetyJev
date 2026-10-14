"""Codex smoke or Jar base sweep; require isolation and audited smoke admission."""
import argparse
import configparser
import fcntl
import hashlib
import importlib.util
import json
import os
import re
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


def verify_smoke(directory, config, source_hashes, config_path, authorized_single=False, queue_update=False, persistent_contacts=False):
    """A longer budget cannot bypass live mechanics and source validation."""
    if directory is None:
        raise ValueError('Jar sweep requires --smoke-evidence')
    plan=json.loads((directory/'plan.json').read_text())
    progress=json.loads((directory/'progress.json').read_text())
    if bool(plan.get('persistent_contacts',False)) != persistent_contacts:
        raise ValueError('Smoke contact-reporting mode differs from sweep')
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
        if queue_update and path == str(Path(__file__).resolve()) and plan['source_sha256'].get(path)==TIMEOUT_QUEUE_SHA256:
            continue  # Only the pinned queue launcher changes; all inference sources/configs match.
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


TIMEOUT_QUEUE_SHA256 = '66fee879bc8d56dda61a94be479964b30887d718a8a05795f079d53bb26e20f4'

TIMEOUT_REASONS = frozenset(('decision_timeout', 'skill_or_planner_timeout'))


def initialization_report(case, scene, seed, wall_seconds, returncode):
    """Accept only a matching simulator declaration of failure before recording/actions."""
    if any((case/'episodes').glob('*/complete.json')):return None
    if any((case/'episodes').rglob('harness-events.jsonl')) or any((case/'episodes').rglob('executed-candidates.jsonl')):
        raise ValueError('Initialization failure has policy/command evidence')
    log=case/'capture.log'
    if not log.exists():return None
    paths=re.findall(r'^Results: (outputs/eval_logs/jar_transport_joint/[^\r\n]+/results\.jsonl)$',log.read_text(errors='replace'),re.M)
    if not paths:return None
    if len(set(paths))!=1:raise ValueError('Ambiguous initialization result')
    base=(Path(jar.MANIGUARD)/'outputs/eval_logs/jar_transport_joint').resolve()
    result_path=(Path(jar.MANIGUARD)/paths[0]).resolve()
    if base not in result_path.parents:raise ValueError('Initialization result outside simulator output')
    values=[json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    if len(values)!=1:raise ValueError('Initialization output must describe exactly one case')
    result=values[0]
    if result.get('status')!='initialization_failed':return None
    cfg_path=result_path.parent/'eval_config.json';cfg=json.loads(cfg_path.read_text())
    if (result.get('scene_name')!=scene or result.get('seed')!=seed or
            cfg.get('scenes')!=[scene] or cfg.get('seed')!=seed or
            cfg.get('tag')!='codex-harness-'+case.name or cfg.get('max_steps')!=2000 or
            type(result.get('steps')) is not int or result['steps']!=0 or
            result.get('success') is not None or result.get('ltl_violated') is not None or
            result.get('counted_violation') is not None or result.get('safety_evaluated') is not False or
            result.get('rollout_diagnostics',{}).get('termination_reason')!='initialization_error'):
        raise ValueError('Initialization failure identity or zero-action contract mismatch')
    return {'status':'initialization_failed','scene':scene,'seed':seed,'executed_steps':0,
        'wall_seconds':wall_seconds,'capture_exit_code':returncode,'result':result,
        'task_success':None,'safe_success':None,'raw_violation':None,
        'failure_reason':'initialization_failed','queue_action':'continue_next_case',
        'execution_audit_passed':False,'initialization_evidence_validated':True,
        'initialization_evidence':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (result_path,cfg_path)},
        'audit_note':'Simulator declares zero policy actions; recorder was not started. No executed-command audit claimed.'}


def validate_queue_update(old, new):
    """Authorize only this queue-only amendment; reject all model/scene/config changes."""
    expected=json.loads(json.dumps(old));path=str(Path(__file__).resolve())
    if expected['source_sha256'].get(path)!=TIMEOUT_QUEUE_SHA256:
        raise ValueError('Queue amendment requires pinned predecessor launcher')
    expected['source_sha256'][path]=new['source_sha256'][path]
    expected['initialization_failure_policy']='record_and_continue'
    if expected!=new:raise ValueError('Queue amendment changed more than initialization continuation')


def terminal_report(case, scene, seed, wall_seconds, returncode, continue_on_timeout, continue_on_initialization_failure=False):
    """Advance only audited timeouts or explicitly validated zero-action initialization failures.

    Missing/corrupt captures, predictor errors, audit failures and unexplained nonzero
    exits remain fatal. Partial actions and raw outcomes stay in the original report.
    """
    eps=list((case/'episodes').glob('*/complete.json'))
    if len(eps)!=1:
        if not eps and continue_on_initialization_failure:
            initial=initialization_report(case,scene,seed,wall_seconds,returncode)
            if initial is not None:return initial
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


def skip_terminal_case(case, continue_on_timeout, continue_on_initialization_failure=False):
    """A restart must not rerun either completed cases or recorded timeout attempts."""
    record=case/'record.json'
    if not record.exists():return False
    old=json.loads(record.read_text())
    if old['status'] in ('failed','initialization_failed') and continue_on_initialization_failure:
        checked=initialization_report(case,old['scene'],old['seed'],old.get('wall_seconds'),old.get('capture_exit_code'))
        if checked is not None:
            if old['status']=='initialization_failed':
                if old['initialization_evidence']!=checked['initialization_evidence']:
                    raise ValueError('Initialization evidence changed')
            else:
                saved=case/'record-before-initialization-classification.json'
                if saved.exists():raise ValueError('Original failure record already archived')
                saved.write_bytes(record.read_bytes());write_json(record,checked)
            return True
        if old['status']=='initialization_failed':raise ValueError('Initialization record failed verification')
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
    if summary['completed']+summary['timeouts']+summary['initialization_failures']!=planned:
        raise RuntimeError('Queue has unprocessed or fatal cases')
    if not summary['timeouts'] and not summary['initialization_failures']:
        return {'status':'completed','completed':planned,'planned':planned}
    if summary['initialization_failures']:
        return {'status':'completed_with_failures','planned':planned,'completed':summary['completed'],
                'timeouts':summary['timeouts'],'initialization_failures':summary['initialization_failures'],
                'attempted':planned,'failed':summary['failed']}
    return {'status':'completed_with_timeouts','planned':planned,'completed':summary['completed'],
            'timeouts':summary['timeouts'],'attempted':planned,'failed':summary['failed']}


def summarize(records, planned):
    done=[r for r in records if r.get('status')=='completed']
    return {'planned':planned,'completed':len(done),
        'failed':sum(r.get('status') in ('failed','infrastructure_failure','timeout','initialization_failed') for r in records),
        'timeouts':sum(r.get('status')=='timeout' for r in records),
        'initialization_failures':sum(r.get('status')=='initialization_failed' for r in records),
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
    p.add_argument('--continue-on-initialization-failure',action='store_true')
    p.add_argument('--persistent-contacts',action='store_true',help='Pinned contact-reporting repair; requires new plan and matching smoke')
    p.add_argument('--queue-update-from',help='SHA256 of the original plan for the authorized queue-only amendment')
    args=p.parse_args();os.umask(0o077)
    config=load_config(args.config)
    if (args.continue_on_initialization_failure or args.queue_update_from) and args.suite!='jar-base':
        raise ValueError('Initialization continuation is for full sweeps only')
    if args.queue_update_from and (not re.fullmatch('[0-9a-f]{64}',args.queue_update_from) or not args.continue_on_initialization_failure):
        raise ValueError('Invalid queue amendment request')
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
    if args.continue_on_initialization_failure:plan['initialization_failure_policy']='record_and_continue'
    if args.persistent_contacts:plan['persistent_contacts']=True
    if args.suite=='jar-base':
        verify_smoke(args.smoke_evidence,config,plan['source_sha256'],args.config,args.authorized_single_smoke,bool(args.queue_update_from),args.persistent_contacts)
        plan['smoke_admission']={'authorized_single':args.authorized_single_smoke,'evidence':str(args.smoke_evidence.resolve())}
    archive=out/'plan-before-initialization-continuation.json'
    if args.queue_update_from:
        source=archive if archive.exists() else out/'plan.json'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=args.queue_update_from:
            raise ValueError('Queue amendment predecessor plan hash mismatch')
        old=json.loads(source.read_text());validate_queue_update(old,plan)
        if not archive.exists():archive.write_bytes(source.read_bytes())
        plan['queue_update']={'previous_plan_sha256':args.queue_update_from,
            'archive':str(archive),'scope':'Queue-only initialization-failure continuation; model/controller/settings unchanged'}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        if not args.queue_update_from or hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()!=args.queue_update_from:
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
        if args.persistent_contacts:command.append('--persistent-contacts')
        if args.continue_on_timeout:command.append('--continue-on-timeout')
        if args.continue_on_initialization_failure:command.append('--continue-on-initialization-failure')
        if args.queue_update_from:command+=['--queue-update-from',args.queue_update_from]
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
            if skip_terminal_case(case,args.continue_on_timeout,args.continue_on_initialization_failure):continue
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
                if args.persistent_contacts:command.insert(command.index('--'), '--persistent-contacts')
                with (case/'capture.log').open('w') as log:
                    capture=subprocess.run(command,cwd=jar.MANIGUARD,env=env,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=cap//8*config['decision_timeout_s']+900)
                report=terminal_report(case,scene,seed,time.monotonic()-started,capture.returncode,args.continue_on_timeout,args.continue_on_initialization_failure)
                write_json(record,report)
                if report['status'] not in ('completed','timeout','initialization_failed'):raise RuntimeError('Harness infrastructure failure')
                if report['status'] in ('timeout','initialization_failed'):
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
