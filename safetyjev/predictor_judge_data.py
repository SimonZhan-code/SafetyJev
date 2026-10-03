"""Build causal per-step forecast indices from complete raw execution records."""
import random
import numpy as np
from .labels import label_forecasts
from .predictor_judge_schema import make_action_window, validate_context, validate_episode_record


def label_predictor_judge_sample(sample, episode):
    return _label_sample(sample, {p['proposal_id']:p for p in episode['proposals']},
                         {e['step']:e for e in episode['execution']}, {r['step']:r for r in episode['oracle']})


def _label_sample(sample, proposals, execution, oracle):
    start,end=sample['start_step'],sample['end_step']
    proposal=proposals[sample['proposal_id']]
    offset=sample['action_offset'];valid=sample['valid_steps']
    actions=proposal['planned_commands'][offset:offset+valid]
    if end-start!=valid or len(actions)!=valid:raise ValueError('Forecast horizon and proposal disagree')
    observed=[oracle[start]] if start in oracle else []
    changed=False;observed_end=start
    for t in range(start,end):
        actual=execution.get(t)
        if actual is None:break
        if (actual['proposal_id']!=sample['proposal_id'] or actual['offset']!=offset+t-start
                or not np.array_equal(np.asarray(actual['command'],dtype=np.float32),np.asarray(actions[t-start],dtype=np.float32))):
            changed=True;break
        row=oracle.get(t+1)
        if row is None or row.get('valid') is not True:break
        observed.append(row);observed_end=t+1
    forecast={'forecast_id':sample['id'],'constraint_id':sample['constraint_id'],
              'start_step':start,'end_step':end,'input':{'remaining_actions':actions}}
    result=label_forecasts([forecast],observed)[0]
    reason=result['label_reason']
    if changed and reason=='censored':reason='action_replaced'
    label=result['label']
    return {'target':None if label is None else ([0.,1.] if label else [1.,0.]),
            'label_reason':reason,'observed_end':observed_end,'first_violation_step':result['first_violation_step']}


def build_predictor_judge_samples(episode, *, history_frames=3, max_actions=8):
    validate_episode_record(episode)
    if type(history_frames) is not int or history_frames<1:raise ValueError('Positive adjacent history length required')
    proposals={p['proposal_id']:p for p in episode['proposals']};rows=[]
    execution={e['step']:e for e in episode['execution']};oracle={r['step']:r for r in episode['oracle']}
    for action in episode['execution']:
        t=action['step'];obs=episode['observations'][t];p=proposals[action['proposal_id']]
        commands=np.asarray(p['planned_commands'],dtype=float)[action['offset']:]
        if not 1<=len(commands)<=max_actions:raise ValueError('Unsupported command horizon')
        valid_steps=len(commands);input_valid=bool(np.isfinite(commands).all())
        history=[s if s>=0 else None for s in range(t-history_frames+1,t+1)]
        for constraint in episode['constraints']:
            cid=constraint['id'];context=obs.get('constraint_context',{}).get(cid,{'source':'none','text':''})
            row={'id':f"{episode['episode_id']}:{t}:{cid}",'episode_id':episode['episode_id'],
                 'group_id':episode['group_id'],'constraint_id':cid,'question':constraint['question'],
                 'requires_history':constraint['requires_history'],
                 'start_step':t,'end_step':t+valid_steps,'valid_steps':valid_steps,
                 'proposal_id':p['proposal_id'],'action_offset':action['offset'],
                 'history_steps':history,'constraint_context':dict(context)}
            row.update(_label_sample(row,proposals,execution,oracle))
            try:validate_context(context,requires_history=constraint['requires_history'])
            except ValueError as exc:
                row.update(target=None,label_reason='context_unavailable',context_error=str(exc))
            if any(not episode['observations'][h].get('valid',True) for h in history if h is not None):
                row.update(target=None,label_reason='invalid_observation')
            if not input_valid:row.update(target=None,label_reason='invalid_actions')
            rows.append(row)
    return rows


def split_groups(samples, *, seed=42, fractions=(.8,.1,.1)):
    fractions=np.asarray(fractions,dtype=float)
    if fractions.shape!=(3,) or not np.isfinite(fractions).all() or (fractions<0).any() or not np.isclose(fractions.sum(),1):
        raise ValueError('Specify train/validation/test fractions summing to one')
    groups=sorted({r['group_id'] for r in samples});random.Random(seed).shuffle(groups)
    n=len(groups);train=int(n*fractions[0]);val=int(n*fractions[1])
    return {g:('train' if i<train else 'validation' if i<train+val else 'test') for i,g in enumerate(groups)}
