"""Validate the transferred runtime, then launch only the 55 reserved Clutter cases."""
import ctypes,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path('/workspace');ART=ROOT/'clutter-bootstrap';STATUS=ART/'startup-status.json'


def save(value):
    p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(STATUS)


def main():
    if os.environ.get('CONTAINER_ID')!='54566989':raise RuntimeError('Wrong Clutter instance')
    receipt=json.loads((ART/'coordinator-reservation-receipt.json').read_text())
    if receipt.get('status')!='clutter_reserved_on_new_node' or receipt.get('clutter_instance')!='54566989':raise RuntimeError('Coordinator reservation not confirmed')
    save({'status':'waiting_for_runtime_transfer'})
    while True:
        try:
            manifest=json.loads((ROOT/'runtime-transfer-manifest.json').read_text())
            assert manifest['source_instance_id']==54533396 and manifest['destination_instance_id']==54566989
            # Isaac rewrites this disposable cache during startup; it is not model or scene data.
            required=[r for r in manifest['files'] if not r['path'].startswith('BEHAVIOR-5.1/OmniGibson/appdata/local/cache/')]
            missing=[r['path'] for r in required if not (ROOT/r['path']).is_file() or (ROOT/r['path']).stat().st_size!=r['bytes']]
            if not missing:break
            save({'status':'waiting_for_runtime_transfer','remaining_files':len(missing),'planned_files':len(manifest['files'])})
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(15)
    save({'status':'validating_runtime'})
    ctypes.CDLL('libGLU.so.1')
    repo=ROOT/'SafetyJev';hashes=json.loads((ART/'expected-source-sha256.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Runtime hash mismatch: '+name)
    py='/workspace/conda/behavior51/bin/python'
    env=dict(os.environ,PYTHONPATH=str(repo)+':/workspace/ManiGuard',SAFETYJEV_MANIGUARD_ROOT='/workspace/ManiGuard')
    subprocess.run([py,'-c','import torch; assert torch.ones(1,device="cuda").item()==1; import spot'],env=env,check=True)
    subprocess.run(['/workspace/openpi/.venv/bin/python','-c','import torch; assert torch.ones(1,device="cuda").item()==1; import openpi'],env=env,check=True)
    subprocess.run([py,'-m','unittest','discover','-s','tests','-q'],cwd=repo,env=env,check=True)
    vulkan=subprocess.run(['vulkaninfo','--summary'],env=dict(os.environ,VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json'),capture_output=True,text=True)
    (ART/'vulkan-summary.txt').write_text(vulkan.stdout+'\n'+vulkan.stderr)
    if vulkan.returncode or 'RTX PRO 6000' not in vulkan.stdout:raise RuntimeError('Vulkan validation failed')
    subprocess.run(['supervisorctl','start','safetyjev-clutter-sweep'],check=True)
    save({'status':'clutter_sweep_started','cases':55,'oracle_only':True,'instance_id':54566989,'unix_s':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise
