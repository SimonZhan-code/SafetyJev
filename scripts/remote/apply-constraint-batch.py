"""One-time, episode-boundary validation and migration from constraint batch 1 to 3."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
NATIVE='/workspace/native-env/bin/python'
SIM='/workspace/conda/behavior51/bin/python'
JOB='safetyjev-jar-agentic-eval'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.umask(0o077)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    run=module('jar_eval',ROOT/'scripts/remote/jar-agentic-eval.py')
    bench=module('constraint_bench',ROOT/'scripts/remote/benchmark-constraint-batch.py')
    root=Path(args.root);config_path=ROOT/'configs/agentic-loop.local.json'
    config=run.load_config(config_path)
    assert config['predictor']['batch_size']==1
    plan=json.loads((root/'plan.json').read_text())
    pause=json.loads((root/'batch-transition-pause.json').read_text())
    pid=pause['pid'];case=Path(pause['case_directory'])
    record_path=case/'record.json';record=json.loads(record_path.read_text())
    case_start=record_path.stat().st_mtime
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    assert b'jar-agentic-eval.py' in command
    status={'status':'waiting_for_current_episode','case':case.name,'old_batch_size':1,'requested_batch_size':3}
    def save(**fields):
        status.update(fields);run.save(out/'transition.json',status)
    save()
    deadline=time.monotonic()+14400
    while True:
        children=subprocess.run(['pgrep','-P',str(pid)],capture_output=True,text=True).stdout.split()
        live=[]
        for child in children:
            stat=subprocess.run(['ps','-o','stat=','-p',child],capture_output=True,text=True).stdout.strip()
            if stat and not stat.startswith('Z'):live.append(child)
        if not live:break
        if time.monotonic()>deadline:
            save(status='blocked',reason='Current episode did not exit before transition deadline')
            raise RuntimeError('Current episode did not finish')
        time.sleep(10)
    # Refuse to overwrite any user changes made while waiting at this boundary.
    assert json.loads((root/'plan.json').read_text())==plan
    assert run.public_config(run.load_config(config_path))==run.public_config(config)
    completed=run.audit(case,record['method'])
    run.save(record_path,{**record,**completed,'wall_seconds':time.time()-case_start,
        'wall_time_source':'record creation to process exit, boundary recovery'})
    # Deliver TERM before CONT so the suspended dispatcher cannot start another case.
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    deadline=time.monotonic()+120
    while (Path('/proc')/str(pid)).exists():
        if time.monotonic()>deadline:raise RuntimeError('Queue dispatcher did not exit')
        time.sleep(1)
    for name in run.SERVICES:run.service('stop',name)
    save(status='benchmarking',preserved_completed=run.summarize(root,plan)['completed'])
    # Keep pi0.5 resident for a realistic model-memory check. Reserve simulator
    # headroom separately because its just-completed process has exited.
    bench_report=out/'benchmark.json'
    candidate=copy.deepcopy(config);candidate['predictor']['batch_size']=3
    applied=False
    try:
        run.service('start','safetyjev-pi05')
        def policy_ready():
            with urlopen('http://127.0.0.1:8000/healthz',timeout=5) as response:
                if response.status!=200:raise RuntimeError('Policy unavailable')
        run.wait_for(policy_ready)
        with (out/'policy-warmup.json').open('w') as log:
            subprocess.run([SIM,str(ROOT/'scripts/remote/warm-jar-policy.py'),'--scene',str(run.BENCH/'task_0000/base')],
                env=run.ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)
        with (out/'benchmark.log').open('w') as log:
            result=subprocess.run([NATIVE,str(ROOT/'scripts/remote/benchmark-constraint-batch.py'),
                '--config',str(config_path),'--root',str(root),'--output',str(bench_report)],
                stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        report=json.loads(bench_report.read_text()) if bench_report.exists() else {}
        if result.returncode or not report.get('passed'):
            raise RuntimeError('Batch benchmark failed acceptance checks')
        total=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        observed=report['summary']['3']['whole_gpu_peak_observed_mib']
        if observed+8192>total*.95:raise RuntimeError('Insufficient simulator memory headroom')
        save(status='checking_http',speedup=report['speedup'],binary_disagreements=report['binary_disagreements'])
        run.save(config_path,candidate,private=True)
        run.service('start','safetyjev-native-server')
        from safetyjev.agentic_predictor import NativePredictor
        health=run.wait_for(NativePredictor(candidate['predictor']).health)
        assert health['batch_size']==3
        probes=[]
        samples=bench.select_samples(root)
        for index in (0,len(samples)-1):
            episode,row,_=samples[index];body=bench.saved_request(episode,row)
            request=Request(candidate['predictor']['endpoint']+'/predict',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=120) as response:answer=json.load(response)
            reference=next(m['answers'] for m in report['samples'][index]['measurements'] if m['batch_size']==3)
            assert all(answer['answers'][qid]['violated']==reference[qid]['violated'] for qid in reference)
            probes.append({'candidate_id':row['candidate_id'],'latency_s':answer['latency_s'],'answers':answer['answers']})
        run.save(out/'http-check.json',{'health':health,'probes':probes})
        changed=copy.deepcopy(plan);changed['config']['predictor']['batch_size']=3
        # No runtime source or scientific parameter besides constraint batching changes.
        expected=json.loads(json.dumps(run.build_plan(candidate,plan['pilot'],plan['all_scenes'],plan['methods'])))
        assert changed==expected
        run.save(root/'plan.before-constraint-batching.json',plan)
        run.save(root/'constraint-batch-migration.json',{
            'old_batch_size':1,'new_batch_size':3,'validation_report':str(bench_report),
            'completed_before_switch':[p.parent.name for p in (root/'cases').glob('*/record.json') if json.loads(p.read_text())['status']=='completed'],
            'changes':'Only constraint batching; candidate generation, budgets, horizon and checkpoint unchanged',
            'numerical_limit':'Zero sampled binary disagreements is not a guarantee on every future input.'})
        run.save(root/'plan.json',changed)
        applied=True
        save(status='validated',benchmark_summary=report['summary'])
    except Exception as exc:
        # Restore the known sequential config on benchmark/OOM/HTTP failure.
        run.save(config_path,config,private=True)
        run.save(root/'plan.json',plan)
        save(status='validation_failed',error_type=type(exc).__name__)
    finally:
        for name in run.SERVICES:run.service('stop',name)
    command=[NATIVE,str(ROOT/'scripts/remote/jar-agentic-eval.py'),'--output',str(root),'--launch','--methods',*plan['methods']]
    if plan['pilot']:command.append('--pilot')
    subprocess.run(command,check=True,capture_output=True,text=True)
    save(status='resumed_batched' if applied else 'resumed_sequential',active_batch_size=3 if applied else 1)
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
