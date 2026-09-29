"""Run serving checks sequentially; the simulator is explicitly absent."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import urlopen

ROOT = Path('/workspace/SafetyJev')
VARIANTS = [
    ('4b-base', 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    ('9b', 'Qwen/Qwen3.5-9B', '47e966881e489511c0c7f5633a9e1960a676a551'),
    ('27b', 'Qwen/Qwen3.8-27B', '28cf73067d5b337860bbef3c85b8b82ba8730956'),
]


def sample_memory(stop, rows):
    while not stop.is_set():
        try:
            value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                             '--format=csv,noheader,nounits'], text=True, timeout=5)
            rows.append({'monotonic_s': time.monotonic(), 'gpu_used_mib': int(value.strip())})
        except Exception as exc:
            rows.append({'error': str(exc)})
        stop.wait(1)


def main():
    subprocess.run(['supervisorctl', 'stop', 'safetyjev-openjev-2b'], check=True)
    summary = []
    for name, model, revision in VARIANTS:
        service = 'safetyjev-openjev-' + name
        print('START', name, flush=True)
        stop = threading.Event()
        memory = []
        thread = threading.Thread(target=sample_memory, args=(stop, memory), daemon=True)
        thread.start()
        record = {'variant': name, 'model_id': model, 'revision': revision,
                  'simulator_executed': False, 'status': 'failed'}
        try:
            subprocess.run(['supervisorctl', 'start', service], check=True, timeout=45)
            deadline = time.monotonic() + 300
            while True:
                try:
                    with urlopen('http://127.0.0.1:8791/health', timeout=3) as response:
                        health = json.load(response)
                    if health['status'] == 'ready' and health['model'] == model:
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Expected model did not become ready')
                time.sleep(2)
            record['health'] = health
            command = ['/workspace/openpi/.venv/bin/python', '-u', str(ROOT / 'remote/joint-model-probe.py'),
                       '--variant', 'openjev-' + name, '--model-id', model, '--revision', revision,
                       '--fixture', str(ROOT / 'artifacts/synthetic-forecast-fixture.json'),
                       '--diagnostics', '/workspace/data/maniguard-bench/jar_transport/task_0000/base/diagnostics.jsonl',
                       '--output', str(ROOT / ('artifacts/joint-model-' + name + '.json'))]
            with (ROOT / ('artifacts/joint-model-' + name + '.log')).open('w') as log:
                subprocess.run(command, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH=str(ROOT)))
            record['status'] = 'passed'
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            subprocess.run(['supervisorctl', 'stop', service], timeout=45)
            stop.set()
            thread.join(timeout=10)
            record['memory_samples'] = memory
            good = [r['gpu_used_mib'] for r in memory if 'gpu_used_mib' in r]
            record['sampled_peak_gpu_used_mib'] = max(good) if good else None
            summary.append(record)
            (ROOT / 'artifacts/model-sweep-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('DONE', name, record['status'], 'sampled_peak_mib', record['sampled_peak_gpu_used_mib'], flush=True)
    if any(r['status'] != 'passed' for r in summary):
        raise SystemExit('Some model sizes failed; see saved summary')


if __name__ == '__main__':
    main()
