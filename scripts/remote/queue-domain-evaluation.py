"""Wait for the active base sweep, then run matched Jar base and every OOD level."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PY='/workspace/conda/behavior51/bin/python'
BASE=Path('/workspace/SafetyJev/artifacts/base-sweep-20261006')
OUT=ROOT/'artifacts/domain-evaluation-20261006'


def save(value):
    OUT.mkdir(parents=True,exist_ok=True)
    temp=OUT/'queue-status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(OUT/'queue-status.json')


def main():
    save({'status':'waiting_for_base','base_root':str(BASE),'planned_base':200,'planned_ood':800})
    print('WAITING for the existing 174-scene base sweep',flush=True)
    while True:
        path=BASE/'progress.json'
        progress=json.loads(path.read_text()) if path.exists() else {}
        result=subprocess.run(['supervisorctl','status','safetyjev-base-sweep'],capture_output=True,text=True)
        if progress.get('status')=='finished' and 'RUNNING' not in result.stdout:break
        if any(state in result.stdout for state in ['FATAL','EXITED']) and progress.get('status')!='finished':
            raise RuntimeError('Base sweep exited before finishing: '+result.stdout.strip())
        time.sleep(30)
    manifest=ROOT/'artifacts/domain-sweep-resources.json'
    resources=json.loads(manifest.read_text())
    if len(resources['families'])!=6 or not all(s.get('ready') for s in resources['families'].values()):
        raise RuntimeError('OOD resource preparation incomplete')
    # Validate every trained scene-specific template against its exact source data.
    import sys,hashlib
    sys.path.insert(0,str(ROOT))
    from safetyjev.visual_runtime import resolve_queries
    checked=0
    for family,spec in resources['families'].items():
        if family=='clutter':continue
        config=json.loads((ROOT/f'configs/{family}-domain-visual.json').read_text())
        source=ROOT/f'configs/visual-queries/{family}.json'
        source_hashes=json.loads(source.read_text()).get('scene_diagnostics_sha256',{}) if source.exists() else {}
        for scenes in spec['scenes_by_level'].values():
            for scene in scenes:
                path=Path('/workspace/data/maniguard-bench')/spec['pipeline']/scene/'diagnostics.jsonl'
                if scene in source_hashes and hashlib.sha256(path.read_bytes()).hexdigest()!=source_hashes[scene]:
                    raise ValueError('Question source mismatch: '+family+'/'+scene)
                diagnostic=json.loads(path.read_text().splitlines()[0])
                resolve_queries(config['queries'],scene,diagnostic['ltl_safety']['propositions']);checked+=1
    print('Validated trained query applicability:',checked,'scenes',flush=True)
    phases=[('jar-base','base',['jar'])]+[(level,level,list(resources['families'])) for level in ['target','language','location','env']]
    phases_done=[]
    for name,level,families in phases:
        run=OUT/name
        if (run/'progress.json').exists() and json.loads((run/'progress.json').read_text()).get('status')=='finished':
            phases_done.append(name);continue
        save({'status':'running','phase':name,'phase_root':str(run),'completed_phases':phases_done,'base_root':str(BASE),'planned_base':200,'planned_ood':800})
        command=[PY,'-u',str(ROOT/'scripts/remote/domain-task-sweep.py'),'--output',str(run),'--level',level,'--families',*families]
        print('START PHASE',name,flush=True)
        result=subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(ROOT)))
        progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
        if progress.get('status')!='finished':raise RuntimeError(f'Phase {name} stopped early (exit {result.returncode})')
        phases_done.append(name);print('FINISHED PHASE',name,progress,flush=True)
    save({'status':'evaluation_finished','completed_phases':phases_done,'base_root':str(BASE),'domain_root':str(OUT),'planned_base':200,'planned_ood':800,'pdf_status':'awaiting_local_compilation_and_visual_review'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc),'base_root':str(BASE)})
        raise
