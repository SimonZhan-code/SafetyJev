"""Resumable Jar ID comparison on the prepared NVIDIA /workspace node.

Run with /workspace/native-env/bin/python. --prepare downloads base scenes;
--launch starts a detached Supervisor job; --check verifies setup without GPU work.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetyjev.agentic_config import load_config, public_config
from safetyjev.agentic_predictor import NativePredictor
from safetyjev.maniguard import verify_sources

BENCH_REV = '2ea32a1451669fb736ae78ffce9cc82aad4cceac'
BENCH = Path('/workspace/data/maniguard-bench/jar_transport')
SIM = '/workspace/conda/behavior51/bin/python'
METHODS = ('baseline', 'local', 'system1')
SERVICES = ('safetyjev-native-server', 'safetyjev-pi05')
JOB = 'safetyjev-jar-agentic-eval'
ENV = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1',
           OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets',
           CUDA_VISIBLE_DEVICES='0', VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
           PYTHONNOUSERSITE='1', PYTHONPATH=str(ROOT)+':/workspace/ManiGuard')


def save(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    tmp.replace(path)
    if private:
        path.chmod(0o600)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def method_config(config, method):
    value = copy.deepcopy(config)
    if method == 'local':
        value['repairs_per_decision'] = 0
        value['system1']['enabled'] = False
        value['system1']['max_calls_per_episode'] = 0
        value['system1']['api_key'] = ''
    return value


def cases(scenes, pilot, methods=METHODS):
    # Pair all three methods at each scene/seed. Policy restarts reset the sampler.
    chosen = scenes[:5] if pilot else scenes
    seeds = (0,) if pilot else (0, 1, 2)
    return [(scene, seed, method) for scene in chosen for seed in seeds for method in methods]


def audit(case_dir, method):
    episodes = list((case_dir/'episodes').glob('*/complete.json'))
    if len(episodes) != 1:
        raise RuntimeError('Expected exactly one terminal episode')
    ep = episodes[0].parent
    complete = json.loads(episodes[0].read_text())
    result = json.loads((ep/'maniguard_result.json').read_text())
    if (complete['status'] != 'completed' or not complete['monitor_valid'] or
            result.get('nan_terminated') or result.get('status') != 'completed' or
            result.get('safetyjev_agentic', {}).get('infrastructure_failure')):
        raise RuntimeError('Episode failed integrity checks')
    selections = rows(ep/'selections.jsonl')
    candidates = rows(ep/'candidate-decisions.jsonl')
    repairs = rows(ep/'planner.jsonl')
    executed = rows(ep/'executed-candidates.jsonl')
    selected = {r['candidate_id']: r for r in selections if 'candidate_id' in r}
    if method != 'baseline':
        if len(executed) != result['steps'] or not executed:
            raise RuntimeError('Missing executed candidate records')
        for record in executed:
            choice = selected[record['candidate_id']]
            offset = record['step']-choice['start_step']-1
            if not (0 <= offset < 8 and record['executed_action'] == choice['commands'][offset]
                    and record['policy_instruction'] == choice['policy_instruction']):
                raise RuntimeError('Selected command/instruction alignment failure')
    return dict(status='completed', episode_id=ep.name, result=result,
                candidates=len(candidates), invalid_candidates=sum(not c['valid'] for c in candidates),
                fallbacks=sum(s.get('fallback', False) for s in selections), repairs=len(repairs),
                repair_errors=sum('error' in p for p in repairs),
                selection_latency_s=[s['selection_latency_s'] for s in selections if 'selection_latency_s' in s])


def summarize(out, plan):
    records = [json.loads(p.read_text()) for p in sorted((out/'cases').glob('*/record.json'))]
    methods = plan.get('methods', METHODS)
    excluded = [r for r in records if r['method'] not in methods]
    records = [r for r in records if r['method'] in methods]
    summary = {'excluded_historical_records':len(excluded), 'planned':len(plan['cases']), 'completed':sum(r['status']=='completed' for r in records),
               'failed':sum(r['status']=='failed' for r in records), 'methods':{}}
    for method in methods:
        good = [r for r in records if r['method']==method and r['status']=='completed']
        per_seed = {}
        for seed in sorted({r['seed'] for r in good}):
            values = [r['result'] for r in good if r['seed']==seed]
            engaged = [r for r in values if r['ever_contacted']]
            per_seed[str(seed)] = {
                'n':len(values), 'success':sum(r['success'] for r in values)/len(values),
                'safe_success':sum(r['success'] and not r['counted_violation'] for r in values)/len(values),
                'engagement':len(engaged)/len(values),
                'counted_violation':sum(r['counted_violation'] for r in values)/len(values),
                'raw_violation':sum(r['ltl_violated'] for r in values)/len(values),
                'engaged_violation':sum(r['counted_violation'] for r in engaged)/len(engaged) if engaged else None}
        summary['methods'][method] = {'per_seed':per_seed, 'completed':len(good),
            'fallbacks':sum(r['fallbacks'] for r in good), 'repairs':sum(r['repairs'] for r in good),
            'repair_errors':sum(r['repair_errors'] for r in good)}
        if per_seed:
            summary['methods'][method]['mean_over_available_seeds'] = {
                k:sum(v[k] for v in per_seed.values())/len(per_seed)
                if all(v[k] is not None for v in per_seed.values()) else None
                for k in ('success','safe_success','engagement','counted_violation','raw_violation','engaged_violation')}
    summary['complete_comparison'] = summary['completed']==summary['planned']
    summary['interpretation'] = 'Rates exclude infrastructure failures; compare paired complete cases. Partial coverage is provisional.'
    save(out/'summary.json', summary)
    return summary


def service(action, name):
    p = subprocess.run(['supervisorctl', action, name], capture_output=True, text=True)
    if p.returncode and not (action=='stop' and 'not running' in p.stdout):
        raise RuntimeError('Supervisor operation failed: '+action+' '+name)


def wait_for(test, seconds=600):
    deadline = time.monotonic()+seconds
    while True:
        try:
            return test()
        except Exception:
            if time.monotonic() >= deadline:
                raise RuntimeError('Service readiness timed out') from None
            time.sleep(3)


def preflight(config):
    verify_sources('/workspace/ManiGuard')
    for path in (SIM, '/workspace/openpi/.venv/bin/python', '/workspace/checkpoints/pi05-jar/7400',
                 config['predictor']['checkpoint_dir']):
        if not Path(path).exists():
            raise RuntimeError('Prepared node is missing '+path)
    if not config['system1']['api_key']:
        raise RuntimeError('OpenRouter key missing from private config')
    if config['predictor']['endpoint'] != 'http://127.0.0.1:8794':
        raise RuntimeError('Prepared service requires predictor port 8794')


def download(config):
    from huggingface_hub import HfApi, snapshot_download
    token = config['predictor']['hf_token'] or None
    info = HfApi(token=token).dataset_info('IDEAS-Lab-Northwestern/ManiGuard-Bench', revision=BENCH_REV)
    files = [s.rfilename for s in info.siblings if s.rfilename.startswith('jar_transport/')
             and '/base/' in s.rfilename and s.rfilename.endswith(('.json', '.jsonl'))]
    scenes = sorted({'/'.join(f.split('/')[1:3]) for f in files})
    if len(scenes) != 26:
        raise RuntimeError('Expected 26 pinned Jar base scenes')
    snapshot_download('IDEAS-Lab-Northwestern/ManiGuard-Bench', repo_type='dataset', revision=BENCH_REV,
                      allow_patterns=files, local_dir=str(BENCH.parent), token=token, max_workers=4)
    return scenes


def build_plan(config, pilot, scenes, methods=METHODS):
    paths = [*sorted((ROOT/'safetyjev').glob('*.py')), Path(__file__),
             ROOT/'scripts/remote/serve-pi05.sh', ROOT/'scripts/remote/warm-jar-policy.py',
             ROOT/'configs/predictor-semantic-definitions.json',
             ROOT/'configs/jar-isaac51-provenance.json',
             Path('/workspace/ManiGuard/configs/eval/jar_transport_joint.yaml')]
    paths += [p for s in scenes for p in sorted((BENCH/s).glob('*.json*'))]
    return {'cases':cases(scenes, pilot, methods), 'methods':list(methods), 'pilot':pilot, 'max_steps':2000,
            'benchmark_revision':BENCH_REV, 'all_scenes':scenes, 'config':public_config(config),
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'limitations':['Experimental Isaac 5.1 parity; selected released policy step is not independently matched to paper.',
                          'Native NVIDIA replay: 1/12 binary disagreements; numerical parity unresolved.',
                          'Predictor semantic targets differ from native temporal predicates.',
                          'Training/evaluation group independence not established.']}


def run(args, config, out, plan):
    records = out/'cases'
    status = subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout
    if any(line.startswith('safetyjev-') and 'RUNNING' in line and not line.startswith(JOB+' ')
           for line in status.splitlines()):
        raise RuntimeError('Another SafetyJev service is running; refusing GPU overlap')
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip():
        raise RuntimeError('GPU has active compute processes; refusing overlap')
    private = ROOT/'configs'
    configs = {}
    for method in METHODS[1:]:
        configs[method] = private/('jar-agentic-'+method+'.local.json')
        save(configs[method], method_config(config, method), private=True)
    provenance = json.loads((ROOT/'configs/jar-isaac51-provenance.json').read_text())
    provenance.update(experiment='jar_agentic_paired', split='base_id_development', pilot=args.pilot)
    save(out/'provenance.json', provenance)
    started_services = set()
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        for scene, seed, method in plan['cases']:
            if build_plan(config, args.pilot, plan['all_scenes'], plan['methods'])['source_sha256'] != plan['source_sha256']:
                raise RuntimeError('Runtime files changed during evaluation')
            key = f"{scene.replace('/', '-')}-seed{seed}-{method}"
            case_dir = records/key
            record_path = case_dir/'record.json'
            identity = dict(scene=scene, seed=seed, method=method)
            if record_path.exists():
                old = json.loads(record_path.read_text())
                if old['status']=='completed':
                    audit(case_dir, method)
                    continue
                if old['status']=='running':
                    try:
                        save(record_path, {**identity, **audit(case_dir, method)})
                        continue
                    except Exception:
                        old.update(status='failed', error='Interrupted run; inspect before explicit retry')
                        save(record_path, old)
                if not args.retry_failed:
                    continue
                archive = case_dir.with_name(key+'-previous-'+str(time.time_ns()))
                (out/'failed-attempts').mkdir(exist_ok=True)
                case_dir.rename(out/'failed-attempts'/archive.name)
            if shutil.disk_usage(out).free < args.min_free_gb*1024**3:
                raise RuntimeError('Disk reserve reached; free space or back up before resuming')
            save(record_path, {**identity, 'status':'running'})
            save(out/'progress.json', {'status':'running','current':identity,**summarize(out,plan)})
            print('START',key,flush=True)
            started = time.monotonic()
            try:
                if method != 'baseline' and SERVICES[0] not in started_services:
                    started_services.add(SERVICES[0]); service('start', SERVICES[0])
                    save(out/'predictor-health.json', wait_for(NativePredictor(config['predictor']).health))
                # Same episode seed must reset even across adjacent paired methods.
                service('stop', SERVICES[1])
                started_services.add(SERVICES[1]); service('start', SERVICES[1])
                def policy_ready():
                    with socket.create_connection(('127.0.0.1',8000),timeout=3):
                        return True
                wait_for(policy_ready)
                with (case_dir/'policy-warmup.json').open('w') as warmup, (case_dir/'warmup.err').open('w') as errors:
                    subprocess.run([SIM,str(ROOT/'scripts/remote/warm-jar-policy.py'),
                                    '--scene',str(BENCH/scene)],env=ENV,stdout=warmup,stderr=errors,
                                   check=True,timeout=360)
                command = [SIM,'-u','-m','safetyjev.visual_runtime' if method=='baseline' else 'safetyjev.cli',
                           'capture','--maniguard-root','/workspace/ManiGuard','--output',str(case_dir/'episodes'),
                           '--provenance',str(out/'provenance.json')]
                command += ['--oracle-only'] if method=='baseline' else ['--agentic-config',str(configs[method])]
                command += ['--','--config','configs/eval/jar_transport_joint.yaml','--benchmark-root',str(BENCH),
                            '--scenes',scene,'--seed',str(seed),'--max-steps','2000','--tag',key]
                with (case_dir/'capture.log').open('w') as log:
                    subprocess.run(command,cwd='/workspace/ManiGuard',env=ENV,stdout=log,
                                   stderr=subprocess.STDOUT,check=True,timeout=args.case_timeout)
                record = {**identity, **audit(case_dir,method), 'wall_seconds':time.monotonic()-started}
            except Exception as exc:
                # No response bodies, configs or potentially credential-bearing exceptions.
                record = {**identity,'status':'failed','error_type':type(exc).__name__,
                          'wall_seconds':time.monotonic()-started}
                save(record_path,record)
                summarize(out,plan)
                raise RuntimeError('Case failed; inspect its capture.log and resume explicitly') from None
            save(record_path,record)
            print('DONE',key,flush=True)
            summarize(out,plan)
        report = summarize(out,plan)
        save(out/'progress.json',{'status':'finished' if report['complete_comparison'] else 'incomplete',**report})
    finally:
        for name in reversed(SERVICES):
            if name in started_services:
                service('stop',name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/agentic-loop.local.json'))
    parser.add_argument('--output',default=str(ROOT/'artifacts/jar-agentic-id'))
    parser.add_argument('--pilot',action='store_true',help='First five sorted scenes, seed 0, all three methods (15 episodes)')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--prepare',action='store_true',help='Download only pinned base scene JSON; no GPU execution')
    parser.add_argument('--check',action='store_true',help='Validate and print plan; no GPU execution')
    parser.add_argument('--launch',action='store_true',help='Launch this queue under Supervisor')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--case-timeout',type=int,default=14400)
    parser.add_argument('--min-free-gb',type=int,default=20)
    args = parser.parse_args()
    os.umask(0o077)
    if len(set(args.methods)) != len(args.methods):
        parser.error('Select each method at most once')
    if args.min_free_gb < 10 or args.case_timeout <= 0:
        parser.error('Require at least 10 GB reserve and positive timeout')
    config = load_config(args.config)
    preflight(config)
    lock = Path('/workspace/safetyjev-jar-agentic.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.prepare:
        scenes = download(config)
    else:
        scenes = sorted(str(p.parent.relative_to(BENCH)) for p in BENCH.glob('task_*/base/diagnostics.jsonl'))
    if len(scenes) != 26 or any(not list((BENCH/s).glob('scene*.json')) for s in scenes):
        raise RuntimeError('Missing base scenes; run --prepare first')
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    plan = json.loads(json.dumps(build_plan(config,args.pilot,scenes,args.methods)))
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text()) != plan:
        raise RuntimeError('Plan/source changed; use a new output directory')
    save(out/'plan.json',plan)
    if args.check or args.prepare:
        print(json.dumps({'status':'ready','episodes':len(plan['cases']),'output':str(out),
                          'services_started':False,'limitations':plan['limitations']},indent=2))
        return
    if args.launch:
        state = subprocess.run(['supervisorctl','status',JOB],capture_output=True,text=True).stdout
        if 'RUNNING' in state or 'STARTING' in state:
            print('Queue is already active'); return
        # Supervisor does not use a shell. Restrict paths so command parsing is unambiguous.
        if any(not all(c.isalnum() or c in '/._-' for c in str(p)) for p in (out,ROOT,Path(args.config).resolve(),Path(sys.executable))):
            raise RuntimeError('Launch paths must not contain spaces or shell metacharacters')
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),
                   '--output',str(out),'--case-timeout',str(args.case_timeout),'--min-free-gb',str(args.min_free_gb)]
        command += ['--methods',*args.methods]
        if args.pilot:command.append('--pilot')
        if args.retry_failed:command.append('--retry-failed')
        Path('/etc/supervisor/conf.d/'+JOB+'.conf').write_text(
            f'[program:{JOB}]\ncommand={" ".join(command)}\ndirectory={ROOT}\n'
            'autostart=false\nautorestart=false\nstopasgroup=true\nkillasgroup=true\nstopwaitsecs=60\n'
            f'stdout_logfile={out}/queue.log\nstderr_logfile={out}/queue.err\n')
        subprocess.run(['supervisorctl','reread'],check=True,capture_output=True)
        subprocess.run(['supervisorctl','update',JOB],check=True,capture_output=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
        service('start',JOB)
        print('Started '+JOB+'; progress: '+str(out/'progress.json'))
        return
    try:
        run(args,config,out,plan)
    except BaseException as exc:
        save(out/'progress.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                 'error_type':type(exc).__name__,**summarize(out,plan)})
        raise


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('Jar evaluation stopped: '+type(exc).__name__+'. Inspect progress.json and case logs.',file=sys.stderr)
        sys.exit(1)
