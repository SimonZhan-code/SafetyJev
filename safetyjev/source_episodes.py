"""Offline adaptation of ManiGuard source episodes; no simulator or oracle model inputs."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .predictor_judge_schema import STATE_FEATURES, validate_episode_record


class MediaReader:
    """Read legacy image files or original JPEG blobs with process-local HDF5 handles."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.handles = {}
        self.pid = os.getpid()

    def read(self, reference, expected=None):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if reference.startswith('hdf5:'):
            import h5py
            _, name, camera, index = reference.split(':')
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or '/' in camera or int(index) < 0:
                raise ValueError('Invalid source image reference')
            if path not in self.handles:
                self.handles[path] = h5py.File(path, 'r')
            group = self.handles[path]['images/' + camera]
            start, end = group['offsets'][int(index):int(index) + 2]
            blob = group['bytes'][int(start):int(end)].tobytes()
        else:
            path = (self.root / reference).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Image path escapes episode')
            blob = path.read_bytes()
        if expected is not None and hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Image bytes changed')
        return blob

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_source_record(directory):
    """Build a small consumer index, retaining the source HDF5 without image extraction.

    Preparation uses ManiGuard's public offline reader. Training consumes only the
    resulting index and HDF5, and does not import ManiGuard, Spot, or Isaac Sim.
    """
    from maniguard.data.recording.reader import EpisodeReader
    from .predictor_judge_observer import base_task_identity, constraint_question

    with EpisodeReader(directory) as reader:
        reader.validate(require_complete=True)
        m = reader.metadata
        cfg = m['configuration']
        if (cfg.get('state_mode') != 'joint' or cfg.get('action_dim') != 8
                or cfg.get('ik_eef_to_joint') or cfg.get('controller_preset') != 'joint_position_raw'
                or not 1 <= cfg.get('execute_horizon', 0) <= 8):
            raise ValueError('Source adapter requires the absolute joint-position policy protocol')
        if m.get('boundaries', 0):
            raise ValueError('Scripted state boundaries are not policy rollout transitions')
        scene = m['scene']; spec = scene['ltl_safety']
        groups = m['joint_groups']; arm = groups['arm']; grip = groups['gripper']
        if len(arm) != 7 or not grip:
            raise ValueError('Unsupported joint-state layout')
        # Use exactly the monitor-resolved objects, rather than guessing glob scope.
        bindings = m['proposition_bindings']
        class Resolver:
            def resolve_patterns(self, patterns):
                result = {}
                for name, definition in spec['propositions'].items():
                    for field in ('over', 'relative_to', 'surface'):
                        declared = definition.get(field, [])
                        declared = [declared] if isinstance(declared, str) else declared
                        if list(patterns) == declared:
                            for obj in bindings[name][field]:
                                result[obj['name']] = SimpleNamespace(**obj)
                if not result:
                    raise ValueError('Missing recorded proposition object bindings')
                return result
        constraints = [dict(id=c['id'], ltl=c['ltl'], requires_history=' U ' in c['ltl'] or ' W ' in c['ltl'],
            question=constraint_question(c, spec, m['constraint_ap_names'][c['id']],
                task_instruction=scene.get('prompt', ''), target_name=scene.get('target_name'), resolver=Resolver()))
            for c in spec['constraints']]
        e = dict(episode_id=m['episode_id'], group_id=base_task_identity(scene), action_dt_s=1/m['action_hz'],
                 state_features=STATE_FEATURES, constraints=constraints, provenance=m.get('provenance', {}),
                 source_metadata=m, result=m['result'], observations=[], proposals=[], execution=[], oracle=[])
        camera_map = {'overview':'image_' + cfg.get('external_cam', 'left'), 'wrist':'wrist_image'}
        for t in range(m['observations']):
            row = reader.read_observation(t); q = row['robot']['q']; dq = row['robot']['dq']
            valid = bool(row['safety']['monitor_valid']) and bool(np.all(row['safety']['ap_valid']))
            clauses = row.get('constraint_records', {})
            if set(clauses) != {c['id'] for c in constraints}:
                raise ValueError('Missing per-constraint GT; source recorder must export clause verdicts')
            values = {cid:bool(value['doomed']) for cid, value in clauses.items()}
            if valid and any(values.values()) != bool(row['safety']['monitor_rejected']):
                raise ValueError('Recorded clause/task monitor disagreement')
            for camera in camera_map.values():
                if camera not in m['camera_keys']:raise ValueError('Required source view is missing')
            e['observations'].append(dict(step=t, valid=valid,
                robot_state=np.r_[q[arm], dq[arm], q[grip].mean(), dq[grip].mean()].astype(float).tolist(),
                images={name:f'hdf5:trajectory.hdf5:{camera}:{t}' for name,camera in camera_map.items()}))
            e['oracle'].append(dict(step=t,valid=valid,violated=values,
                ap=dict(zip(m['ap_names'],map(bool,row['safety']['ap_values'])))))
        for event in reader.iter_events():
            if event['type'] != 'proposal':continue
            raw = np.asarray(event['actions'], dtype=np.float32)
            planned = raw[:event['planned_steps']].copy()
            if planned.ndim != 2 or planned.shape[1] != 8:raise ValueError('Unsupported proposal shape')
            if cfg.get('gripper_binarize'):
                planned[:, -1] = np.where(np.abs(planned[:, -1]) > .01, np.sign(planned[:, -1]), -1)
            low, high = event.get('action_low'), event.get('action_high')
            if low is None or high is None:raise ValueError('Missing action-space limits')
            planned = np.clip(planned, np.asarray(low,dtype=np.float32), np.asarray(high,dtype=np.float32)).astype(np.float32)
            e['proposals'].append(dict(proposal_id=str(event['proposal_id']), start_step=event['source_observation_id'],
                planned_commands=planned.tolist(), planned_steps=len(planned), raw_actions=raw.tolist()))
        for t in range(m['transitions']):
            tr = reader.read_transition(t)
            if not tr.get('issued', True) or tr.get('source_boundary_id', -1) != -1:
                raise ValueError('Non-policy execution cannot label the proposed action sequence')
            if not np.isclose(tr['duration_s'], e['action_dt_s']):raise ValueError('Action clock mismatch')
            e['execution'].append(dict(step=t, proposal_id=str(tr['proposal_id']), offset=int(tr['proposal_offset']),
                                      command=tr['applied'].tolist()))
        validate_episode_record(e)
        return e
