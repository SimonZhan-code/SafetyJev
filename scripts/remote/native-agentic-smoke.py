"""One base Jar integration smoke; services must already be healthy on localhost."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from safetyjev.agentic_config import load_config, public_config
from safetyjev.agentic_predictor import NativePredictor
from safetyjev.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-steps', type=int, default=256)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_steps % 8:
        parser.error('max-steps must be a positive multiple of eight')
    root = Path('/workspace/SafetyJev')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    write_json(output/'predictor-health.json', NativePredictor(config['predictor']).health())
    write_json(output/'plan.json', {'scene':'task_0000/base', 'seed':0, 'max_steps':args.max_steps,
        'config':public_config(config), 'scope':'One bounded Jar integration smoke, not a benchmark or safety-improvement claim'})
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1',
        OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets', CUDA_VISIBLE_DEVICES='0',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', PYTHONNOUSERSITE='1',
        PYTHONPATH=str(root)+':/workspace/ManiGuard')
    stopped = threading.Event()
    def sample_gpu():
        with (output/'gpu.csv').open('w') as stream:
            stream.write('unix_s,memory_mib,utilization_percent\n')
            while not stopped.is_set():
                result = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True,text=True)
                stream.write(str(time.time())+','+result.stdout);stream.flush();stopped.wait(1)
    sampler = threading.Thread(target=sample_gpu,daemon=True);sampler.start()
    started = time.monotonic()
    try:
        with (output/'capture.log').open('w') as log:
            run = subprocess.run(['/workspace/conda/behavior51/bin/python','-u','-m','safetyjev.cli','capture',
                '--maniguard-root','/workspace/ManiGuard','--output',str(output/'episodes'),
                '--provenance',str(root/'configs/jar-isaac51-provenance.json'),
                '--agentic-config',str(Path(args.config).resolve()), '--',
                '--config','configs/eval/jar_transport_joint.yaml',
                '--benchmark-root','/workspace/data/maniguard-bench/jar_transport',
                '--scenes','task_0000/base','--seed','0','--max-steps',str(args.max_steps),
                '--tag','safetyjev-native-agentic'],cwd='/workspace/ManiGuard',env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        episodes = list((output/'episodes').glob('*/complete.json'))
        if run.returncode or len(episodes) != 1:
            write_json(output/'summary.json',{'status':'infrastructure_failure','process_exit':run.returncode,
                'completed_episode_records':len(episodes),'wall_seconds':time.monotonic()-started})
            raise RuntimeError('Capture failed; inspect capture.log')
        episode = episodes[0].parent
        complete = json.loads(episodes[0].read_text())
        result = json.loads((episode/'maniguard_result.json').read_text())
        def rows(name):
            file = episode/name
            return read_jsonl(file) if file.exists() else []
        selections = rows('selections.jsonl'); candidates = rows('candidate-decisions.jsonl')
        executed = rows('executed-candidates.jsonl'); repairs = rows('planner.jsonl')
        selected = {c['candidate_id']:c for c in selections if 'candidate_id' in c}
        alignment = all(r['candidate_id'] in selected and
            r['executed_action'] == selected[r['candidate_id']]['commands'][r['step']-selected[r['candidate_id']]['start_step']-1] and
            r['policy_instruction'] == selected[r['candidate_id']]['policy_instruction'] for r in executed)
        valid = complete['status']=='completed' and complete['monitor_valid'] and alignment and bool(executed)
        report = {'status':'passed' if valid else 'infrastructure_failure','wall_seconds':time.monotonic()-started,
            'episode_id':episode.name,'complete':complete,'result':result,
            'candidate_attempts':len(candidates),'invalid_candidates':sum(not c['valid'] for c in candidates),
            'accepted_selections':sum(not c['fallback'] for c in selections if 'fallback' in c),
            'fallback_selections':sum(c['fallback'] for c in selections if 'fallback' in c),
            'repair_calls':len(repairs),'successful_repairs':sum('vla_instruction' in p for p in repairs),
            'selected_command_and_instruction_alignment':alignment,
            'executed_steps':len(executed),
            'scope':'Integration smoke; ManiGuard outcomes are distinct from trained semantic targets'}
        write_json(output/'summary.json',report)
        print(json.dumps(report),flush=True)
        if not valid:raise RuntimeError('Completed capture failed integration audit')
    finally:
        stopped.set();sampler.join(timeout=5)


if __name__=='__main__':main()
