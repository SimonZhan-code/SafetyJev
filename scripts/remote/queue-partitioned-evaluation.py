"""Execute only one worker's immutable, disjoint assignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from partitioned_plan import validate_assignment

ROOT = Path(__file__).resolve().parents[2]
PY = '/workspace/conda/behavior51/bin/python'
BASE = Path('/workspace/SafetyJev/artifacts/base-sweep-20261006')
OUT = ROOT / 'artifacts/domain-evaluation-20261006'


def save(value):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / 'queue-status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(OUT / 'queue-status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['node-a', 'node-b'], required=True)
    args = parser.parse_args()
    path = ROOT / 'configs/two-node-assignments.json'
    assignment = json.loads(path.read_text())
    resources = json.loads((ROOT / 'artifacts/domain-sweep-resources.json').read_text())
    validate_assignment(assignment, resources)
    worker = assignment['workers'][args.worker]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    common = {'worker':args.worker, 'assignment_file_sha256':digest,
              'owned_cases':len(worker['cases']), 'scope':'200 base + 800 OOD across both workers',
              'base_pdf':'deliver after all 200 base cases; do not wait for OOD'}
    if worker['wait_for_original_base']:
        save(dict(common, status='waiting_for_base'))
        while True:
            progress = json.loads((BASE / 'progress.json').read_text())
            state = subprocess.run(['supervisorctl','status','safetyjev-base-sweep'], capture_output=True,text=True)
            if progress.get('status') == 'finished' and 'RUNNING' not in state.stdout:
                break
            if any(s in state.stdout for s in ['FATAL','EXITED','STOPPED']) and progress.get('status') != 'finished':
                raise RuntimeError('Original base process stopped before completion: '+state.stdout)
            time.sleep(30)
    # Verify all scene-specific trained questions before running any assigned phase.
    import sys
    sys.path.insert(0, str(ROOT))
    from safetyjev.visual_runtime import resolve_queries
    for family, spec in resources['families'].items():
        if family == 'clutter':
            continue
        config = json.loads((ROOT / f'configs/{family}-domain-visual.json').read_text())
        source = ROOT / f'configs/visual-queries/{family}.json'
        hashes = json.loads(source.read_text()).get('scene_diagnostics_sha256', {}) if source.exists() else {}
        for scenes in spec['scenes_by_level'].values():
            for scene in scenes:
                diagnostic = Path('/workspace/data/maniguard-bench') / spec['pipeline'] / scene / 'diagnostics.jsonl'
                if scene in hashes and hashlib.sha256(diagnostic.read_bytes()).hexdigest() != hashes[scene]:
                    raise ValueError('Question diagnostic source mismatch: '+family+'/'+scene)
                resolve_queries(config['queries'], scene, json.loads(diagnostic.read_text().splitlines()[0])['ltl_safety']['propositions'])
    finished = []
    for phase in worker['phases']:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Assignment changed while queue was active')
        run = OUT / phase['name']
        if (run / 'progress.json').exists() and json.loads((run / 'progress.json').read_text()).get('status') == 'finished':
            if args.worker=='node-b' and phase['name']=='jar-base':
                subprocess.run([PY,str(ROOT/'scripts/remote/sync-jar-base.py')],check=True)
            finished.append(phase['name'])
            continue
        save(dict(common, status='running', phase=phase['name'], completed_phases=finished))
        command = [PY,'-u',str(ROOT / 'scripts/remote/domain-task-sweep.py'),
                   '--output',str(run),'--level',phase['level'],'--families',*phase['families'],
                   '--assignment',str(path),'--worker',args.worker,'--fail-fast']
        if phase['reverse_scenes']:
            command.append('--reverse-scenes')
        result = subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        progress = json.loads((run / 'progress.json').read_text()) if (run / 'progress.json').exists() else {}
        if progress.get('status') != 'finished':
            raise RuntimeError(f"Phase {phase['name']} stopped early: exit {result.returncode}")
        if args.worker=='node-b' and phase['name']=='jar-base':
            subprocess.run([PY,str(ROOT/'scripts/remote/sync-jar-base.py')],check=True)
        finished.append(phase['name'])
        save(dict(common, status='between_phases', completed_phases=finished))
    save(dict(common, status='worker_finished', completed_phases=finished))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise
