"""Compare native constraint batch sizes on saved Jar requests, without simulation.

Stop the native HTTP service first: this process loads the same pinned model.
No VLA sampling, simulator actions, or OpenRouter calls are performed.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from safetyjev.agentic_config import load_config
from safetyjev.agentic_predictor import request_digest


def saved_request(episode, row):
    inputs = copy.deepcopy(row['inputs'])
    valid = sum(inputs['history_mask'])
    history = row['history_images'][-valid:]
    if len(history) != valid:
        raise ValueError('Missing saved history')
    inputs['observations'] = {camera:[None]*(8-valid)+[
        base64.b64encode((episode/item['images'][camera]).read_bytes()).decode() for item in history]
        for camera in ('overview','wrist')}
    if request_digest(row['questions'],inputs) != row['request_sha256']:
        raise ValueError('Reconstructed request digest mismatch')
    return {'request_id':row['candidate_id'],'questions':row['questions'],'inputs':inputs}


def select_samples(root):
    complete = []
    for record in sorted((root/'cases').glob('*/record.json')):
        data=json.loads(record.read_text())
        if data['method']=='system1' and data['status']=='completed':
            complete.append((record.parent/'episodes'/data['episode_id'],data))
    if len(complete)<4:raise ValueError('At least four completed episodes required')
    # Even coverage across the completed scene/seed sequence, selected before inference.
    indices=sorted({round(i*(len(complete)-1)/4) for i in range(5)})
    samples=[]
    for index in indices:
        episode,record=complete[index]
        rows=[json.loads(line) for line in (episode/'candidate-inputs.jsonl').read_text().splitlines()]
        for j in sorted({0, min(1,len(rows)-1),len(rows)//2,len(rows)-1}):
            row=rows[j]
            if len(row['questions'])!=3:raise ValueError('Expected three Jar constraints')
            samples.append((episode,row,{'scene':record['scene'],'seed':record['seed'],
                'episode_id':record['episode_id'],'candidate_id':row['candidate_id'],
                'step':row['step'],'request_sha256':row['request_sha256']}))
    return samples


def gpu_memory_mib():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return int(r.stdout.strip().splitlines()[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise ValueError('Use a fresh report path')
    if args.repeats<2:raise ValueError('At least two repeats required')
    output.parent.mkdir(parents=True,exist_ok=True)
    samples=select_samples(Path(args.root))
    # Confirm exact request reconstruction before allocating a model.
    for episode,row,_ in samples:saved_request(episode,row)
    import torch
    from safetyjev.agentic_server import NativeService
    config=load_config(args.config)
    service=NativeService(config)
    report={'status':'running','samples':[], 'repeats':args.repeats,
        'identity':service.health, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'measurement':'Single loaded model; whole predict() wall time includes decoding; no HTTP transport. Alternate order across samples/repeats.',
        'co_resident_services':subprocess.run(['supervisorctl','status','safetyjev-pi05'],capture_output=True,text=True).stdout.strip()}
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output)
    save()
    # Warm both batch shapes on a full-history request, outside measurements.
    full=next((s for s in samples if all(s[1]['inputs']['history_mask'])),samples[0])
    for batch in (1,3):
        service.batch_size=batch
        service.predict(saved_request(full[0],full[1]))
    for index,(episode,row,meta) in enumerate(samples):
        record={**meta,'measurements':[]}
        body=saved_request(episode,row)
        for repeat in range(args.repeats):
            for batch in ((1,3) if (index+repeat)%2==0 else (3,1)):
                service.batch_size=batch
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter();answer=service.predict(body);torch.cuda.synchronize()
                record['measurements'].append({'batch_size':batch,'repeat':repeat,
                    'wall_s':time.perf_counter()-start,'model_region_s':answer['latency_s'],
                    'answers':answer['answers'],
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                    'whole_gpu_mib_after':gpu_memory_mib()})
        reference=record['measurements'][0]['answers']
        record['binary_disagreements_vs_reference']=sum(
            reference[q]['violated']!=m['answers'][q]['violated']
            for m in record['measurements'][1:] for q in reference)
        record['max_abs_logit_delta']=max(abs(reference[q]['logit']-m['answers'][q]['logit'])
            for m in record['measurements'] for q in reference)
        report['samples'].append(record);save()
        print(json.dumps({'sample':index+1,'total':len(samples),'binary_disagreements':record['binary_disagreements_vs_reference']}),flush=True)
    measurements=[m for r in report['samples'] for m in r['measurements']]
    summary={}
    for batch in (1,3):
        ms=[m for m in measurements if m['batch_size']==batch]
        summary[str(batch)]={'mean_wall_s':statistics.mean(m['wall_s'] for m in ms),
            'median_wall_s':statistics.median(m['wall_s'] for m in ms),
            'peak_allocated_gib':max(m['peak_allocated_gib'] for m in ms),
            'peak_reserved_gib':max(m['peak_reserved_gib'] for m in ms),
            'whole_gpu_peak_observed_mib':max(m['whole_gpu_mib_after'] for m in ms)}
    report.update(status='completed',summary=summary,
        speedup=summary['1']['mean_wall_s']/summary['3']['mean_wall_s'],
        binary_disagreements=sum(r['binary_disagreements_vs_reference'] for r in report['samples']),
        max_abs_logit_delta=max(r['max_abs_logit_delta'] for r in report['samples']))
    report['passed']=report['binary_disagreements']==0 and report['speedup']>1
    save();print(json.dumps({k:report[k] for k in ['status','summary','speedup','binary_disagreements','max_abs_logit_delta','passed']},indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
