"""Audit completed compatibility pilot artifacts; never supplies labels to models."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0,'/workspace/SafetyJev')
from safetyjev.labels import label_forecasts
from safetyjev.metrics import evaluate,quantile

ROOT=Path('/workspace/SafetyJev/artifacts')
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def episode(root):
 paths=list(root.glob('*/complete.json'))
 assert len(paths)==1,(root,paths)
 return paths[0].parent
base=episode(ROOT/'jar-580-2b-isaac51')
forecasts=rows(base/'forecasts.jsonl');oracle=rows(base/'oracle.jsonl')
assert len(oracle)==2001 and all(x['valid'] for x in oracle)
assert len(forecasts)==1000
labels=label_forecasts(forecasts,oracle)
eligible={x['forecast_id'] for x in labels if x['label'] is not None}
assert len(eligible)==283
sweep=read(ROOT/'joint-rollout-sweep.json')
assert len(sweep)==3 and all(x['status']=='passed' for x in sweep)
records={x['variant']:x for x in sweep}
memory=[]
for values in csv.reader((ROOT/'gpu-memory-580.csv').read_text().splitlines()):
 if len(values)!=3: continue
 stamp=datetime.strptime(values[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
 memory.append((stamp,int(values[1]),int(values[2])))
summary={'scope':'Isaac 5.1 development compatibility pilot; not a benchmark-equivalent accuracy validation','full_episode_id':base.name,'full_episode_steps':2000,'forecasts':1000,'eligible_including_global':283,'per_constraint_positive_events':2,'global_positive_windows':1,'task_result':read(base/'maniguard_result.json'),'models':[]}
manifest={}
for variant in ['2b','4b-base','9b','27b']:
 ep=base if variant=='2b' else episode(ROOT/('jar-580-'+variant+'-isaac51'))
 online=rows(ep/'predictions.jsonl');truth=rows(ep/'oracle.jsonl');complete=read(ep/'complete.json')
 assert all(x['valid'] for x in truth)
 assert complete['final_step']==(2000 if variant=='2b' else 64)
 assert len(online)==(1000 if variant=='2b' else 32)
 assert all(x.get('error') is None and x.get('score') is not None for x in online)
 pred=online if variant=='2b' else rows(base/('replay-'+variant+'.jsonl'))
 if variant!='2b': assert {x['forecast_id'] for x in pred}==eligible
 assert all(x.get('error') is None and x.get('score') is not None for x in pred)
 nonglobal=[x for x in labels if x['constraint_id']!='__all__'];ids={x['forecast_id'] for x in nonglobal}
 metrics=evaluate(nonglobal,[x for x in pred if x['forecast_id'] in ids])
 assert metrics['prediction_coverage']==1 and metrics['missing_predictions']==metrics['failed_predictions']==0
 if variant=='2b':
  start=datetime(2026,9,29,5,50,9,tzinfo=timezone.utc).timestamp();end=(base/'complete.json').stat().st_mtime
 else:
  start=records[variant]['joint_started_unix_s'];end=records[variant]['joint_ended_unix_s']
 used=[m[1] for m in memory if start<=m[0]<=end]
 assert used
 timings=[p['latency_s'] for p in online]
 row={'variant':variant,'joint_steps':complete['final_step'],'joint_predictions':len(online),'joint_latency_p50_ms':1000*quantile(timings,.5),'joint_latency_p95_ms':1000*quantile(timings,.95),'joint_sampled_peak_gpu_mib':max(used),'memory_sample_count':len(used),'joint_start_unix_s':start,'joint_end_unix_s':end,'per_constraint_metrics_on_same_inputs':metrics['micro'],'replay_prediction_count':len(pred)}
 summary['models'].append(row)
 for file in ep.glob('*.json*'):
  manifest[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
for file in base.glob('*.json*'):manifest[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
(ROOT/'compatibility-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'capture-manifest-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['variant','joint_steps','joint_predictions','joint_latency_p50_ms','joint_sampled_peak_gpu_mib']} for r in summary['models']],indent=2))
