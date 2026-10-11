"""Causal input contracts for per-action violation prediction."""
import math
import numpy as np

STATE_FEATURES = ([f'arm_q_{i}' for i in range(7)] + [f'arm_qdot_{i}' for i in range(7)]
                  + ['gripper_q', 'gripper_qdot'])
CAMERAS = ('overview', 'wrist')


def make_action_window(commands, offset, *, max_actions=8, dt_s=.05):
    commands = np.array(commands, dtype=np.float32, copy=True)
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError('max_actions must be positive')
    if (commands.ndim != 2 or commands.shape[1] != 8 or
            not 0 < len(commands) <= max_actions or not np.isfinite(commands).all()):
        raise ValueError('Expected finite planned commands with shape (1..max_actions,8)')
    if type(offset) is not int or not 0 <= offset < len(commands):
        raise ValueError('offset must select an unexecuted command')
    if isinstance(dt_s, bool) or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Positive finite command interval required')
    valid = len(commands)-offset
    actions = np.zeros((max_actions, 8), dtype=np.float32)
    actions[:valid] = commands[offset:]
    times = np.zeros(max_actions, dtype=np.float32)
    times[:valid] = np.arange(valid, dtype=np.float32)*dt_s
    return {'actions':actions, 'action_mask':np.arange(max_actions)<valid,
            'valid_steps':valid, 'relative_times_s':times, 'action_dt_s':float(dt_s)}


def validate_context(context, *, requires_history):
    if not isinstance(context, dict) or set(context) != {'source','text'}:
        raise ValueError('Context requires explicit source and text')
    if context['source'] not in ('none','observed_history') or not isinstance(context['text'],str):
        raise ValueError('Only disclosed observed-history context is supported; oracle is not model input')
    if context['source']=='none' and context['text']:
        raise ValueError('Absent context cannot contain text')
    if type(requires_history) is not bool:
        raise ValueError('Declare history requirements as a Boolean')
    if context['source']=='observed_history' and not context['text'].strip():
        raise ValueError('Observed context must contain text')
    # History-dependent rules remain supervised from the full oracle trace.
    # Missing explicit memory is not a reason to drop a short-window example.


def validate_episode_record(record):
    for key in ('episode_id','group_id'):
        if not isinstance(record.get(key),str) or not record[key]:raise ValueError('Missing '+key)
    dt=record['action_dt_s']
    if isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:raise ValueError('Invalid control clock')
    if record['state_features'] != STATE_FEATURES:raise ValueError('Unsupported robot state schema')
    observations=record['observations']
    if not observations:raise ValueError('Missing initial observation')
    for t,obs in enumerate(observations):
        state=np.asarray(obs['robot_state'],dtype=float)
        if type(obs['step']) is not int or obs['step']!=t:raise ValueError('Observation timeline has a gap/duplicate')
        if state.shape!=(16,) or (obs.get('valid',True) and not np.isfinite(state).all()):raise ValueError('Invalid robot state')
        if set(obs['images'])!=set(CAMERAS):raise ValueError('Missing camera view')
    proposals={}
    for p in record['proposals']:
        pid=p['proposal_id'];start=p['start_step']
        if not isinstance(pid,str) or not pid or pid in proposals:raise ValueError('Duplicate/invalid proposal ID')
        if type(start) is not int or not 0<=start<len(observations):raise ValueError('Invalid proposal time')
        planned=np.asarray(p['planned_commands'],dtype=float)
        if planned.ndim!=2 or planned.shape[1]!=8 or not 1<=len(planned)<=8:raise ValueError('Invalid planned command shape')
        if p['planned_steps']!=len(p['planned_commands']):raise ValueError('Plan length mismatch')
        raw=np.asarray(p['raw_actions'],dtype=float)
        if raw.ndim!=2 or raw.shape[1]!=8 or len(raw)<p['planned_steps']:
            raise ValueError('Invalid full policy proposal')
        proposals[pid]=p
    if len(record['execution'])!=len(observations)-1:
        partial=record.get('result',{}).get('status') in ('crashed','numerical_failed','monitor_failed')
        if not partial or len(record['execution'])!=len(observations):raise ValueError('T actions require T+1 observations, except an explicitly failed final observation')
    for t,e in enumerate(record['execution']):
        p=proposals.get(e['proposal_id']);offset=e['offset']
        if e['step']!=t or type(e['step']) is not int or p is None:raise ValueError('Invalid execution identity')
        if type(offset) is not int or not 0<=offset<p['planned_steps'] or p['start_step']+offset!=t:
            raise ValueError('Invalid execution offset')
        if not np.array_equal(np.asarray(e['command'],dtype=np.float32),np.asarray(p['planned_commands'][offset],dtype=np.float32)):
            raise ValueError('Executed command differs from committed proposal')
    ids=[c['id'] for c in record['constraints']]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Constraint IDs must be unique')
    for c in record['constraints']:
        if not isinstance(c['question'],str) or not c['question'].strip():raise ValueError('Missing natural-language constraint')
        if type(c['requires_history']) is not bool:raise ValueError('Declare history requirements')
    seen=set()
    for row in record['oracle']:
        t=row['step']
        if type(t) is not int or not 0<=t<len(observations) or t in seen:raise ValueError('Invalid oracle timeline')
        seen.add(t)
