"""Node-specific real rollout smoke checks plus identical-input model replay.

Requires the completed Isaac 5.1 compatibility pilot and model services. This is
not a held-out accuracy benchmark. Each follow-up rollout has a 64-step limit.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT=Path('/workspace/SafetyJev')
ART=ROOT/'artifacts'
PYTHON='/workspace/conda/behavior51/bin/python'
BASE=ART/'jar-580-2b-isaac51'
VARIANTS=[('4b-base','Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),('9b','Qwen/Qwen3.5-9B','47e966881e489511c0c7f5633a9e1960a676a551'),('27b','Qwen/Qwen3.8-27B','28cf73067d5b337860bbef3c85b8b82ba8730956')]
ENV=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_HEADLESS='1',OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets',CUDA_VISIBLE_DEVICES='0',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',PYTHONNOUSERSITE='1',PYTHONPATH='/workspace/SafetyJev:/workspace/ManiGuard')

def run_cli(args,log,timeout=1800):
 with open('/workspace/logs/'+log,'w') as f:
  subprocess.run([PYTHON,'-u','-m','safetyjev.cli']+args,cwd='/workspace/ManiGuard',env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def main():
 assert list(BASE.glob('*/complete.json')), '2B pilot must complete before this sweep'
 records=[]
 subprocess.run(['supervisorctl','stop','safetyjev-openjev-2b'],check=True)
 for variant,model,rev in VARIANTS:
  service='safetyjev-openjev-'+variant
  record={'variant':variant,'model_id':model,'revision':rev,'started_unix_s':time.time(),'status':'failed','joint_rollout_max_steps':64}
  print('START',variant,flush=True)
  try:
   subprocess.run(['supervisorctl','start',service],check=True,timeout=45)
   deadline=time.monotonic()+360
   while True:
    try:
     with urlopen('http://127.0.0.1:8791/health',timeout=3) as r: health=json.load(r)
     if health.get('status')=='ready' and health.get('model')==model: break
    except Exception: pass
    if time.monotonic()>deadline: raise TimeoutError('Model did not become ready')
    time.sleep(2)
   config=ART/('predictor-'+variant+'.json')
   config.write_text(json.dumps({'endpoint':'http://127.0.0.1:8791/v1/systemone','model_id':model,'predictor_revision':rev,'input_mode':'proprio_only','timeout':30}))
   output=ART/('jar-580-'+variant+'-isaac51')
   subprocess.run([PYTHON,str(ROOT/'scripts/remote/reset-policy.py')],env=ENV,check=True,timeout=180)
   record['joint_started_unix_s']=time.time()
   run_cli(['capture','--maniguard-root','/workspace/ManiGuard','--output',str(output),'--provenance',str(ROOT/'configs/jar-isaac51-provenance.json'),'--online-predictor',str(config),'--','--config','configs/eval/jar_transport_joint.yaml','--benchmark-root','/workspace/data/maniguard-bench/jar_transport','--scenes','task_0000/base','--seed','0','--max-steps','64','--tag','safetyjev-580-'+variant],'capture-580-'+variant+'.log')
   record['joint_ended_unix_s']=time.time()
   completed=list(output.glob('*/complete.json'))
   if len(completed)!=1: raise RuntimeError('Expected one complete real rollout')
   episode=completed[0].parent
   forecasts=[json.loads(line) for line in (episode/'forecasts.jsonl').read_text().splitlines()]
   predictions=[json.loads(line) for line in (episode/'predictions.jsonl').read_text().splitlines()]
   oracle=[json.loads(line) for line in (episode/'oracle.jsonl').read_text().splitlines()]
   if not forecasts or len(predictions)!=len(forecasts) or any(p.get('error') is not None for p in predictions): raise RuntimeError('Missing or failed online predictions')
   if any(not r['valid'] for r in oracle): raise RuntimeError('Invalid oracle labels')
   record['joint_prediction_count']=len(predictions)
   record['joint_last_step']=oracle[-1]['step']
   record['joint_status']='passed'
   run_cli(['report','--episodes',str(output),'--output',str(ART/('joint-report-'+variant+'.json'))],'joint-report-'+variant+'.log')
   name='replay-'+variant
   record['replay_started_unix_s']=time.time()
   run_cli(['predict','--episodes',str(BASE),'--name',name,'--endpoint','http://127.0.0.1:8791/v1/systemone','--model-id',model,'--predictor-revision',rev,'--input-mode','proprio_only','--eligible-only'],'replay-'+variant+'.log',timeout=2400)
   run_cli(['report','--episodes',str(BASE),'--predictions',name,'--output',str(ART/('replay-report-'+variant+'.json'))],'replay-report-'+variant+'.log')
   replay_report=json.loads((ART/('replay-report-'+variant+'.json')).read_text())
   if replay_report['prediction_coverage'] != 1 or replay_report['failed_predictions'] or replay_report['missing_predictions']:
    raise RuntimeError('Incomplete replay coverage')
   record['status']='passed'
  except Exception as e:
   record['error']=type(e).__name__+': '+str(e)
  finally:
   subprocess.run(['supervisorctl','stop',service],timeout=60)
   record['ended_unix_s']=time.time()
   records.append(record)
   (ART/'joint-rollout-sweep.json').write_text(json.dumps(records,indent=2)+'\n')
  print('DONE',variant,record,flush=True)
 if any(r['status']!='passed' for r in records): raise SystemExit(1)

if __name__=='__main__': main()
