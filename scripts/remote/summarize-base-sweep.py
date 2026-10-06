"""Summarize planned case coverage and family reports, including running sweeps."""
import argparse
import json
from pathlib import Path


def summarize(root):
    plan=json.loads((root/'plan.json').read_text());families={}
    for family,spec in plan['families'].items():
        scenes=spec['scenes'][:plan['max_scenes_override']] if plan['max_scenes_override'] else spec['scenes']
        cases=[]
        for scene in scenes:
            path=root/family/'cases'/(scene.replace('/','-')+'.json')
            cases.append(json.loads(path.read_text()) if path.exists() else {'scene':scene,'status':'pending'})
        completed=[c for c in cases if c['status']=='completed']
        counts={status:sum(c['status']==status for c in cases) for status in ['pending','running','completed','failed']}
        entry={'planned':len(scenes),**counts,'task_successes':sum(c['result']['success'] for c in completed),
               'contacted':sum(c['result']['ever_contacted'] for c in completed),
               'grasped':sum(c['result']['ever_grasped'] for c in completed),
               'executed_actions':sum(c['result']['steps'] for c in completed),
               'raw_ltl_violations':sum(bool(c['result'].get('ltl_violated')) for c in completed),
               'counted_violations':sum(bool(c['result'].get('counted_violation')) for c in completed),
               'failures':[{'scene':c['scene'],'error':c.get('error')} for c in cases if c['status']=='failed'],
               'classifier':'not requested' if family=='clutter' else 'trained step-20000 current predicates'}
        report=root/family/'report.json'
        if report.exists():
            d=json.loads(report.read_text());entry['classification']={k:d[k] for k in ['expected_classifications','failed_or_missing_classifications','coverage','frame_latency_s','raw','calibrated']}
        audit=root/family/'audit-status.json'
        if audit.exists():entry['trace_audit_passed']=json.loads(audit.read_text())['passed']
        families[family]=entry
    result={'scope':plan['scope'],'complete':all(not f['pending'] and not f['running'] for f in families.values()),
            'planned':sum(f['planned'] for f in families.values()),'completed':sum(f['completed'] for f in families.values()),
            'failed':sum(f['failed'] for f in families.values()),'families':families,
            'limitations':['Experimental Isaac 5.1 compatibility stack; physical label parity unverified.',
                           'Base split does not establish independence from SafetyJev training groups.',
                           'Synchronous current-state classification, not future action-conditioned prediction.',
                           'Clutter has no classifier scores by explicit user choice.']}
    temp=root/'summary.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(root/'summary.json')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    r=summarize(args.root);print(json.dumps({k:r[k] for k in ['complete','planned','completed','failed']}))
