"""Causal chunk-boundary numeric inputs shared by recorded and runtime checks."""
import numpy as np
from .predictor_judge_schema import make_action_window

CHUNK_CONTRACT='chunk_start_v2'


def chunk_numeric_inputs(record,row):
    if row.get('input_contract')!=CHUNK_CONTRACT or row['action_offset']!=0:
        raise ValueError('Expected chunk-start input contract')
    t=row['start_step']
    proposal=next(p for p in record['proposals'] if p['proposal_id']==row['proposal_id'])
    if proposal['start_step']!=t or len(proposal['planned_commands'])!=8:
        raise ValueError('Expected eight commands committed at this boundary')
    future=make_action_window(proposal['planned_commands'],0,dt_s=record['action_dt_s'])
    state=np.asarray(record['observations'][t]['robot_state'],dtype=np.float32)
    if state.shape!=(16,) or not np.isfinite(state).all():raise ValueError('Invalid current robot state')
    past=np.zeros((8,8),dtype=np.float32)
    states=np.zeros((8,16),dtype=np.float32)
    steps=row['history_action_steps'];history=row['history_steps']
    if len(steps)!=8 or len(history)!=8:raise ValueError('Eight aligned history slots required')
    valid=[s is not None for s in steps]
    if valid!=[False]*(8-sum(valid))+[True]*sum(valid):raise ValueError('History actions must be an adjacent suffix')
    for i,s in enumerate(steps):
        if s is not None:
            if s!=t-8+i or history[i]!=s+1 or s<0:
                raise ValueError('Historical action/result observation misalignment')
            transition=record['execution'][s]
            if transition['step']!=s:raise ValueError('Missing historical transition')
            past[i]=transition['command']
            result_state=np.asarray(record['observations'][s+1]['robot_state'],dtype=np.float32)
            if result_state.shape!=(16,) or not np.isfinite(result_state).all():raise ValueError('Invalid historical robot state')
            states[i]=result_state
    if not np.isfinite(past).all():raise ValueError('Nonfinite executed action')
    return dict(robot_state=state,history_robot_states=states,
        remaining_actions=future['actions'],action_mask=future['action_mask'],action_dt_s=future['action_dt_s'],
        history_mask=np.asarray([s is not None for s in history],dtype=bool),
        executed_actions=past,history_action_mask=np.asarray(valid,dtype=bool))
