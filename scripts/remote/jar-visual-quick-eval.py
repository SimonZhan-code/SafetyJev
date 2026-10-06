"""Three base Jar scenes, one seed, 256 actions each, current visual AP scoring.

Assumes the documented /workspace layout and localhost policy/classifier servers.
No guard intervention, OOD scenes, or OpenRouter calls. Refuses to overwrite runs.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import urlopen

ROOT=Path('/workspace/SafetyJev')
ART=ROOT/'artifacts/jar-visual-step20000-20261006'
PY='/workspace/conda/behavior51/bin/python'
ENV=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_HEADLESS='1',
         OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets',
         CUDA_VISIBLE_DEVICES='0',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
         PYTHONNOUSERSITE='1',PYTHONPATH='/workspace/SafetyJev:/workspace/ManiGuard')
SCENES=['task_0000/base','task_0001/base','task_0002/base']


def main():
    ART.mkdir(parents=True,exist_ok=False)
    (ART/'plan.json').write_text(json.dumps({'scenes':SCENES,'seed':0,'max_steps':256,
       'sample_stride':8,'threshold':.5,'execution':'unchanged_pi05',
       'checkpoint_selection':'released final step-20000, chosen before new rollout results',
       'split_note':'base-only development evaluation; training-group overlap not yet verified'},indent=2)+'\n')
    with urlopen('http://127.0.0.1:8792/health',timeout=10) as r:
        (ART/'classifier-health.json').write_bytes(r.read())
    stop=threading.Event()
    def sample():
        with (ART/'gpu.csv').open('w') as out:
            out.write('unix_s,memory_used_mib,utilization_percent\n')
            while not stop.is_set():
                p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
                out.write(str(time.time())+','+p.stdout);out.flush();stop.wait(1)
    worker=threading.Thread(target=sample,daemon=True);worker.start()
    records=[]
    try:
        for scene in SCENES:
            record={'scene':scene,'start_unix_s':time.time(),'status':'failed'}
            print('START',scene,flush=True)
            try:
                subprocess.run([PY,str(ROOT/'scripts/remote/reset-policy.py')],env=ENV,check=True,timeout=240)
                previous=set((ART/'episodes').glob('*/episode.json'))
                with open('/workspace/logs/visual-'+scene.split('/')[0]+'.log','w') as log:
                    subprocess.run([PY,'-u','-m','safetyjev.visual_runtime','capture',
                        '--maniguard-root','/workspace/ManiGuard','--output',str(ART/'episodes'),
                        '--provenance',str(ROOT/'configs/jar-isaac51-provenance.json'),
                        '--classifier-config',str(ROOT/'configs/jar-visual-step20000.json'),
                        '--','--config','configs/eval/jar_transport_joint.yaml',
                        '--benchmark-root','/workspace/data/maniguard-bench/jar_transport',
                        '--scenes',scene,'--seed','0','--max-steps','256','--tag','safetyjev-visual-step20000'],
                        cwd='/workspace/ManiGuard',env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
                added=set((ART/'episodes').glob('*/episode.json'))-previous
                if len(added)!=1:raise RuntimeError('Expected one captured episode')
                episode=next(iter(added)).parent
                complete=json.loads((episode/'complete.json').read_text())
                result=json.loads((episode/'maniguard_result.json').read_text())
                if complete['status']!='completed' or not complete['monitor_valid']:raise RuntimeError('Invalid completed episode')
                if result['safetyjev_classification']['failed_samples']:raise RuntimeError('Classifier request failures')
                record.update(status='completed',episode_id=episode.name,result=result)
            except Exception as exc:record['error']=type(exc).__name__+': '+str(exc)
            record['end_unix_s']=time.time();records.append(record)
            (ART/'sweep.json').write_text(json.dumps(records,indent=2)+'\n')
            print('DONE',scene,record['status'],record.get('error',''),flush=True)
        subprocess.run([PY,'-m','safetyjev.visual_runtime','report','--episodes',str(ART/'episodes'),
                        '--output',str(ART/'report.json')],env=ENV,cwd=ROOT,check=True)
    finally:
        stop.set();worker.join(timeout=5)
    if any(r['status']!='completed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
