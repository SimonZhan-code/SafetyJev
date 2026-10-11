"""Native chunk_start_v2 history and static trained-query bindings."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .semantic_labels import _scoped_subject


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def robot_state(robot):
    arm = array(robot.arm_control_idx[robot.default_arm]).astype(int)
    grip = array(robot.gripper_control_idx[robot.default_arm]).astype(int)
    q, v = array(robot.get_joint_positions()), array(robot.get_joint_velocities())
    if len(arm) != 7 or not len(grip):
        raise ValueError('Expected seven arm joints and a nonempty gripper')
    state = np.r_[q[arm], v[arm], q[grip].mean(), v[grip].mean()].astype(np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError('Invalid native robot state')
    return state.tolist()


def controller_commands(chunk, action_space):
    low, high = array(action_space.low), array(action_space.high)
    if low.shape != (8,) or high.shape != (8,) or not (low[-1] <= -1 and high[-1] >= 1):
        raise ValueError('Expected eight controller bounds including a binary gripper')
    if np.isnan(low).any() or np.isnan(high).any() or (low > high).any():
        raise ValueError('Invalid controller bounds')
    commands = np.array(chunk, dtype=np.float32, copy=True)
    if commands.ndim != 2 or commands.shape[1] != 8 or len(commands) < 8:
        raise ValueError('Native predictor requires eight future actions')
    commands = commands[:8].copy()
    if not np.isfinite(commands).all():
        raise ValueError('Nonfinite VLA actions')
    g = commands[:, -1]
    commands[:, -1] = np.where(np.abs(g) > .01, np.sign(g), -1.)
    commands = np.clip(commands, action_space.low, action_space.high)
    if commands.shape != (8, 8) or not np.isfinite(commands).all():
        raise ValueError('Invalid controller bounds')
    return commands


def bind_queries(scene, resolver, required_aps, selected=None):
    raw = (Path(__file__).parents[1]/'configs/predictor-semantic-definitions.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '89853b4eb5eb49d83c84a5a0de2b7e473d510a25db36076d4f48502b1f3ba5f9':
        raise ValueError('Published semantic definitions have changed')
    definitions = json.loads(raw)
    family = scene.get('pipeline') or Path(scene['scene_file']).parent.parent.parent.name
    spec = scene['ltl_safety']['propositions']
    bindings = {}
    for name, prop in spec.items():
        patterns = prop.get('over', [])
        if isinstance(patterns, str):
            patterns = [patterns]
        objects = resolver.resolve_patterns(patterns).values() if patterns else []
        bindings[name] = {'over': [{'name': o.name, 'category': getattr(o, 'category', 'object')} for o in objects]}
    evidence = {'bindings': bindings, 'task_instruction': scene.get('prompt', ''),
                'target_name': scene.get('target_name')}
    result, covered = [], set()
    for definition in definitions['definitions']:
        names = definition['scope_bindings'].get(family, [])
        if not names:
            continue
        conjunction = definition['label_rule']['kind'] == 'ap_all_of'
        for group in ([names] if conjunction else [[name] for name in names]):
            if not all(name in spec for name in group):
                continue
            objects = bindings[group[0]]['over']
            if not objects:
                raise ValueError('Empty trained object scope: ' + group[0])
            subject, instruction = _scoped_subject(objects, group[0], evidence)
            question = definition['question'].format(subject=subject)
            if instruction:
                question = 'Task for identifying the objects: ' + instruction + '\n' + question
            sid = definition['semantic_id']
            qid = sid if conjunction else sid + ':' + group[0]
            result.append({'id': qid, 'question': 'During execution of the remaining planned actions: ' + question,
                           'semantic_id': sid, 'source_aps': group, 'objects': objects,
                           'label_rule': copy.deepcopy(definition['label_rule'])})
            covered.update(group)
    if not result or set(required_aps)-covered:
        raise ValueError('Task contains unsupported native predictor constraints: ' + repr(sorted(set(required_aps)-covered)))
    if selected is not None:
        if set(selected)-{q['id'] for q in result}:
            raise ValueError('Requested query does not apply to this scene')
        result = [q for q in result if q['id'] in selected]
    return result


class ChunkHistory:
    def __init__(self):
        self.rows = []

    def append(self, step, state, images, command=None):
        if (not self.rows and step != 0) or (self.rows and step != self.rows[-1]['step']+1):
            raise ValueError('Native history has a step gap')
        if (step == 0) != (command is None):
            raise ValueError('History requires an executed action for every resulting state')
        self.rows.append({'step': step, 'state': list(state), 'images': dict(images),
                          'command': None if command is None else list(command)})
        self.rows = self.rows[-8:]

    def inputs(self, commands, dt, directory):
        if not self.rows:
            raise ValueError('No current native observation')
        # Match training: initial observation is shown alone; later slots pair
        # each executed action with its resulting image/state, excluding o_0.
        rows = [r for r in self.rows if r['command'] is not None] or [self.rows[-1]]
        slots = [None]*(8-len(rows)) + rows
        observations = {name: [] for name in ('overview', 'wrist')}
        for row in slots:
            for camera in observations:
                observations[camera].append(None if row is None else base64.b64encode(
                    (Path(directory)/row['images'][camera]).read_bytes()).decode())
        return {'observations': observations, 'robot_state': self.rows[-1]['state'],
                'history_robot_states': [r['state'] if r and r['command'] is not None else [0.]*16 for r in slots],
                'executed_actions': [r['command'] if r and r['command'] is not None else [0.]*8 for r in slots],
                'history_mask': [r is not None for r in slots],
                'history_action_mask': [r is not None and r['command'] is not None for r in slots],
                'remaining_actions': np.asarray(commands).tolist(), 'action_mask': [True]*8,
                'action_dt_s': dt, 'constraint_context': {'source': 'none', 'text': ''}}
