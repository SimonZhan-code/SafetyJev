"""Simulator-process snapshots for recorded-command replay, not policy-server replay."""
import copy,inspect,random
import numpy as np


def predicate_memory(monitor):
    return {name:copy.deepcopy(inspect.signature(fn).parameters['_state'].default)
            for name,fn in monitor._prop_fns.items() if '_state' in inspect.signature(fn).parameters}


def capture_snapshot(env,monitor):
    import torch
    protected=[]
    # Upstream AG dump_state converts a nested position in a shallow-copied dict.
    # Give serialization its own copy so the live Python constraint is not changed.
    for robot in env.robots:
        if hasattr(robot,'_ag_obj_constraint_params'):
            original=robot._ag_obj_constraint_params;protected.append((robot,original))
            robot._ag_obj_constraint_params=copy.deepcopy(original)
    try:state=copy.deepcopy(env.scene.dump_state(serialized=False))
    finally:
        for robot,original in protected:robot._ag_obj_constraint_params=original
    return {'scene':state,'monitor':{'state':monitor._monitor.state if hasattr(monitor._monitor,'state') else monitor._monitor._state,
            'violation_step':monitor._violation_step,'violation_count':monitor._violation_count,
            'error':copy.deepcopy(monitor._error),'last_log':copy.deepcopy(monitor._ltl_log[-1:]),
            'predicate_memory':predicate_memory(monitor)},
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                   'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            'scope':'scene+robot/controller/AG+particles+monitor+simulator-process RNG; replay recorded commands; external policy-server RNG not captured'}


def restore_snapshot(env,monitor,snapshot,*,convert_state=None):
    import torch
    if convert_state is None:
        from omnigibson.utils.python_utils import recursively_convert_to_torch
        convert_state=recursively_convert_to_torch
    env.scene.load_state(convert_state(copy.deepcopy(snapshot['scene'])),serialized=False)
    saved=snapshot['monitor'];monitor._monitor._state=saved['state']
    monitor._violation_step=saved['violation_step'];monitor._violation_count=saved['violation_count']
    monitor._error=copy.deepcopy(saved['error']);monitor._ltl_log=copy.deepcopy(saved['last_log'])
    for name,value in saved['predicate_memory'].items():
        state=inspect.signature(monitor._prop_fns[name]).parameters['_state'].default
        state.clear();state.update(copy.deepcopy(value))
    def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,(tuple,list)) else x
    rng=snapshot['rng'];random.setstate(tuples(rng['python']))
    nr=rng['numpy'];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),int(nr[2]),int(nr[3]),float(nr[4])))
    torch.set_rng_state(torch.as_tensor(rng['torch'],dtype=torch.uint8,device='cpu'))
    if rng['cuda']:
        if len(rng['cuda'])!=torch.cuda.device_count():raise ValueError('Snapshot CUDA device count mismatch')
        torch.cuda.set_rng_state_all([torch.as_tensor(v,dtype=torch.uint8,device='cpu') for v in rng['cuda']])
