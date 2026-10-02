"""Episode audit and train-only membership; raw records are never rewritten."""
from collections import defaultdict
import math
import json
import random
import numpy as np
from .predictor_judge_schema import validate_episode_record


def inventory_episode(record, *, active_motion_rad=.05):
    """Safety describes the observed trace; quality separately controls admission."""
    if not math.isfinite(active_motion_rad) or active_motion_rad<0:
        raise ValueError('Finite nonnegative active motion threshold required')
    reasons=[]
    try:
        validate_episode_record(record)
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        reasons.append('invalid_record: '+str(exc))
    observations=record.get('observations',[]);execution=record.get('execution',[])
    constraints=[c['id'] for c in record.get('constraints',[])]
    oracle=record.get('oracle',[]);events=[];first={}
    for row in sorted(oracle,key=lambda r:r.get('step',-1)):
        if row.get('valid') is not True:continue
        for cid in constraints:
            if row.get('violated',{}).get(cid) is True and cid not in first:
                first[cid]=row['step'];events.append({'constraint_id':cid,'step':row['step']})
    expected=list(range(len(observations)))
    if ([r.get('step') for r in oracle]!=expected or not expected or not constraints
            or any(r.get('valid') is not True or any(type(r.get('violated',{}).get(c)) is not bool for c in constraints) for r in oracle)):
        reasons.append('incomplete_or_invalid_monitor')
    if any(e['step']==0 for e in events):reasons.append('initially_violated')
    result=record.get('result',{})
    if result.get('status')!='completed' or result.get('nan_terminated'):
        reasons.append('incomplete_rollout')
    if len(execution)!=len(observations)-1 or not execution:reasons.append('incomplete_execution')
    motion=None
    try:
        states=np.asarray([o['robot_state'] for o in observations],dtype=float)
        if states.ndim!=2 or states.shape[1]!=16 or not np.isfinite(states).all() or any(o.get('valid',True) is not True for o in observations):
            raise ValueError('invalid state')
        motion=float(np.ptp(states[:,:7],axis=0).max())
    except (ValueError,KeyError,TypeError):reasons.append('invalid_observation')
    source=record.get('provenance',{}).get('checkpoint',{})
    checkpoint={k:source.get(k) for k in ('repo','revision','step')}
    checkpoint_id=json.dumps(checkpoint,sort_keys=True,separators=(',',':'))
    return {'checkpoint':checkpoint,'checkpoint_id':checkpoint_id,'episode_id':record.get('episode_id'),'group_id':record.get('group_id'),
            'family':str(record.get('group_id','unknown')).split('/')[0],
            'policy':record.get('provenance',{}).get('checkpoint',{}).get('repo','unknown'),
            'steps':len(execution),'safety':'unsafe' if events else ('unknown' if reasons else 'safe'),
            'quality_ok':not reasons,'quality_reasons':reasons,'events':events,
            'arm_motion_rad':motion,'active':motion is not None and motion>active_motion_rad,
            'active_motion_threshold_rad':active_motion_rad}


def select_episodes(inventory, assignments, *, seed=42, unsafe_per_safe=4):
    if not math.isfinite(unsafe_per_safe) or unsafe_per_safe<=0:
        raise ValueError('Positive unsafe_per_safe required')
    rows=sorted((dict(r) for r in inventory),key=lambda r:r['episode_id'])
    if len({r['episode_id'] for r in rows})!=len(rows):raise ValueError('Duplicate episode identity')
    if any(assignments.get(r['group_id']) not in ('train','validation','test') for r in rows):
        raise ValueError('Frozen split must cover every episode group')
    by_family=defaultdict(list)
    for r in rows:
        r.update(split=assignments[r['group_id']],selected=False,selection_reason='quality_excluded')
        if not r['quality_ok']:continue
        if r['split']!='train':r.update(selected=True,selection_reason='heldout_complete')
        elif r['safety']=='unsafe':r.update(selected=True,selection_reason='unsafe_training_core')
        elif not r['active']:r['selection_reason']='inactive_safe'
        else:r['selection_reason']='safe_training_quota'
        if r['split']=='train':by_family[r['family']].append(r)
    rng=random.Random(seed)
    for family in sorted(by_family):
        candidates=by_family[family];unsafe=sum(r['safety']=='unsafe' for r in candidates)
        quota=max(1,math.ceil(unsafe/unsafe_per_safe))
        # Round-robin checkpoint sources avoids selecting all safe examples from one policy.
        safe=defaultdict(list)
        for r in candidates:
            if r['safety']=='safe' and r['active']:safe[r['checkpoint_id']].append(r)
        for bucket in safe.values():rng.shuffle(bucket)
        policies=sorted(safe);rng.shuffle(policies)
        while quota and any(safe.values()):
            for policy in policies:
                if quota and safe[policy]:
                    r=safe[policy].pop();r.update(selected=True,selection_reason='active_safe_supplement');quota-=1
    return rows


def episode_counts(rows):
    from collections import Counter
    return {'total':len(rows),'safety':dict(Counter(r['safety'] for r in rows)),
            'quality_ok':sum(r['quality_ok'] for r in rows),
            'selection_reasons':dict(Counter(r['selection_reason'] for r in rows))}


def composition_report(inventory, windows):
    selected=[r for r in inventory if r['selected']]
    report={'episodes':{'all':episode_counts(inventory),'selected':episode_counts(selected),
                        'by_split':{},'by_family':{},'by_policy':{},'by_checkpoint_id':{}},
            'events':{'all':sum(len(r['events']) for r in inventory),'selected':sum(len(r['events']) for r in selected),
                      'definition':'distinct episode/constraint/first-rejection step; correlated constraints may share one physical incident'},
            'windows':windows,'training_coverage':[]}
    for field in ['split','family','policy','checkpoint_id']:
        for value in sorted({str(r[field]) for r in inventory}):
            pool=[r for r in inventory if str(r[field])==value]
            report['episodes']['by_'+field][value]={'all':episode_counts(pool),'selected':episode_counts([r for r in pool if r['selected']])}
    for field in ['split','family','policy','checkpoint_id','constraint']:
        counts={}
        for r in inventory:
            values=[r['family']+'/'+e['constraint_id'] for e in r['events']] if field=='constraint' else [str(r[field])]*len(r['events'])
            if field!='constraint':counts.setdefault(str(r[field]),{'all':0,'selected':0})
            for value in values:
                item=counts.setdefault(value,{'all':0,'selected':0})
                item['all']+=1;item['selected']+=int(r['selected'])
        if field=='constraint':
            for key in windows.get('by_constraint',{}):counts.setdefault(key,{'all':0,'selected':0})
        report['events']['by_'+field]=counts
    for fam in sorted({r['family'] for r in inventory if r['split']=='train'}):
        pool=[r for r in selected if r['split']=='train' and r['family']==fam]
        unsafe=sum(r['safety']=='unsafe' for r in pool);safe=sum(r['safety']=='safe' for r in pool)
        report['training_coverage'].append({'family':fam,'unsafe':unsafe,'safe':safe,
             'unsafe_fraction':unsafe/len(pool) if pool else None,'needs_unsafe_collection':unsafe==0})
    return report


def write_summary(path, report):
    lines=['# Predictor Judge data package','',
           'Safety labels describe the captured horizon. Training membership is curated; validation/test are not outcome-balanced.',
           'Raw files are referenced with relative paths and are required alongside this package. No raw episodes were deleted.','',
           '| Scope | Episodes | Unsafe | Safe | Unknown |','|---|---:|---:|---:|---:|']
    for key in ['all','selected']:
        r=report['episodes'][key];s=r['safety'];lines.append(f"| {key} | {r['total']} | {s.get('unsafe',0)} | {s.get('safe',0)} | {s.get('unknown',0)} |")
    lines+=['','| Split | Positive windows | Negative windows | Excluded windows |','|---|---:|---:|---:|']
    for split,r in sorted(report['windows']['by_split'].items()):
        lines.append(f"| {split} | {r.get('positive',0)} | {r.get('negative',0)} | {r.get('excluded',0)} |")
    lines+=['',f"Selected first-rejection events: {report['events']['selected']}. {report['events']['definition']}.",'',
            'Full episode inventory: `episode_inventory.jsonl`. Counts by family, checkpoint, constraint, remaining length and exclusion reason: `composition.json`.',
            'The training pool is not duplicated to balance labels. Actual epoch draws, unique samples and repetition are reported separately by the training sampler.',
            'A family without positive events requires additional training collection; duplicated windows do not supply new events.']
    path.write_text('\n'.join(lines)+'\n')
