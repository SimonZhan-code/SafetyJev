"""Three 64-action development cases on the configured PRO 6000 node.

Threshold 0.35 is a deliberate rejection-path diagnostic, not a tuned threshold.
Uses real π0.5, Open-Jev 2B, Isaac 5.1, and (planner case) OpenRouter.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import threading

ROOT = Path('/workspace/SafetyJev')
ART = ROOT / 'artifacts/planner-integration'
PYTHON = '/workspace/conda/behavior51/bin/python'
ENV = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1',
           OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets',
           CUDA_VISIBLE_DEVICES='0', VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
           PYTHONNOUSERSITE='1', PYTHONPATH='/workspace/SafetyJev:/workspace/ManiGuard')
if not ENV.get('OPENROUTER_API_KEY'):
    ENV['OPENROUTER_API_KEY'] = Path('/run/safetyjev/openrouter.key').read_text().strip()
CASES = [('shadow', []),
         ('guard', ['--execution-mode', 'guard_regenerate', '--guard-threshold', '0.35', '--max-regenerations', '3']),
         ('planner', ['--execution-mode', 'guard_regenerate', '--guard-threshold', '0.35', '--max-regenerations', '3',
                      '--planner-config', str(ROOT / 'configs/openrouter-deepseek-v4.1-flash.json')])]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def sample_gpu(stop):
    with (ART/'gpu-memory.csv').open('w') as output:
        output.write('timestamp,memory_used_mib,utilization_percent\n')
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            output.write(result.stdout)
            output.flush()
            stop.wait(1)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop,), daemon=True)
    sampler.start()
    records = []
    for name, extra in CASES:
        output = ART / name
        if output.exists():
            raise ValueError('Refusing to overwrite case ' + name)
        record = {'case': name, 'started_unix_s': time.time(), 'max_steps': 64,
                  'threshold': None if name == 'shadow' else .35, 'status': 'failed'}
        print('START', name, flush=True)
        try:
            # The native server reseeds only when episode_seed changes.
            subprocess.run([PYTHON, str(ROOT/'scripts/remote/reset-policy.py')], env=ENV,
                           cwd='/workspace/ManiGuard', check=True, timeout=240)
            command = [PYTHON, '-u', '-m', 'safetyjev.cli', 'capture',
                       '--maniguard-root', '/workspace/ManiGuard', '--output', str(output),
                       '--provenance', str(ROOT/'configs/jar-isaac51-provenance.json'),
                       '--online-predictor', str(ROOT/'configs/openjev-state-only.json')] + extra + [
                       '--', '--config', 'configs/eval/jar_transport_joint.yaml',
                       '--benchmark-root', '/workspace/data/maniguard-bench/jar_transport',
                       '--scenes', 'task_0000/base', '--seed', '0', '--max-steps', '64',
                       '--tag', 'safetyjev-planner-integration-' + name]
            with open('/workspace/logs/integration-' + name + '.log', 'w') as log:
                subprocess.run(command, env=ENV, cwd='/workspace/ManiGuard', stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            complete = list(output.glob('*/complete.json'))
            if len(complete) != 1:
                raise RuntimeError('Expected one completed captured episode')
            episode = complete[0].parent
            result = json.loads((episode/'maniguard_result.json').read_text())
            oracle = rows(episode/'oracle.jsonl')
            if not oracle or any(not r['valid'] for r in oracle):
                raise RuntimeError('Missing or invalid oracle coverage')
            if result['status'] != 'completed':
                raise RuntimeError('ManiGuard reported ' + result['status'])
            plans = rows(episode/'planner.jsonl')
            decisions = rows(episode/'decisions.jsonl')
            forecasts = rows(episode/'forecasts.jsonl')
            predictions = rows(episode/'predictions.jsonl')
            if len(predictions) != len(forecasts) or any(r.get('error') for r in predictions):
                raise RuntimeError('Missing or failed selected-candidate scores')
            executed_replacements = [r for r in decisions if r.get('decision') == 'accept'
                                     and r['attempt'] > 0 and r['start_step'] < result['steps']]
            record.update(status='passed', episode_id=episode.name, result=result,
                          oracle_samples=len(oracle), forecasts=len(forecasts),
                          planner_calls=len(plans), successful_planner_calls=sum('vla_instruction' in r for r in plans),
                          executed_replacement_chunks=len(executed_replacements))
            subprocess.run([PYTHON, '-m', 'safetyjev.cli', 'report', '--episodes', str(output),
                            '--threshold', str(.5 if name == 'shadow' else .35),
                            '--output', str(ART/(name+'-report.json'))], env=ENV, check=True, timeout=60)
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        record['ended_unix_s'] = time.time()
        records.append(record)
        (ART/'sweep.json').write_text(json.dumps(records, indent=2)+'\n')
        print('DONE', name, record['status'], record.get('error', ''), flush=True)
    stop.set()
    sampler.join(timeout=5)
    if any(r['status'] != 'passed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
