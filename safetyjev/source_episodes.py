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


def _evidence_value(value=None, *, reason=None):
    """JSON-safe measured value; unknown evidence must never turn into a No."""
    if reason is not None:
        return dict(available=False, value=None, reason=reason)

    def convert(item):
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError('nonfinite_measurement')
        return item

    try:
        return dict(available=True, value=convert(value), reason=None)
    except ValueError as exc:
        return dict(available=False, value=None, reason=str(exc))


def _source_measurement(record):
    if not isinstance(record, dict):
        return _evidence_value(reason='measurement_not_recorded')
    if not record.get('valid'):
        return _evidence_value(reason=record.get('error') or 'invalid_source_measurement')
    if 'value' not in record or record['value'] is None:
        return _evidence_value(reason='measurement_not_recorded')
    return _evidence_value(record['value'])


def extract_semantic_evidence(directory, *, observation_ids=None, liquid_volumes=None):
    """Read physical evidence without rendering, relabeling or copying source media.

    observation_ids optionally bounds a development audit; returned source counts
    still describe the full episode. Source hashes cover small metadata/scene
    files; evidence_sha256 covers the extracted values, not RGB or entire HDF5.
    Snapshot particle totals are diagnostic only, never container occupancy.
    The recorded world surface top supplies a fixed work-surface reference plane.
    Optional liquid_volumes maps exact runtime object names to validated rigid
    container volumes. Missing geometry stays unavailable, never total particles.
    Measured fields remain supervision-side evidence, not a model input payload.
    Static task instruction/target identity can ground ambiguous query scopes;
    resolved IDs and measured state labels are never copied into the question.
    """
    import json
    from maniguard.data.recording.reader import EpisodeReader

    directory = Path(directory)
    with EpisodeReader(directory) as reader:
        reader.validate(require_complete=True, decode_images=False)
        m = reader.metadata
        if m.get('boundaries', 0):
            raise ValueError('Semantic extraction requires policy observations without scripted boundaries')
        n = m['observations']
        ids = list(range(n)) if observation_ids is None else list(observation_ids)
        if (not ids or any(type(i) is not int or not 0 <= i < n for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError('Observation indices must be unique, increasing and in range')
        names = m.get('object_names', [])
        if len(names) != len(set(names)):
            raise ValueError('Duplicate source object names')
        bindings = m.get('proposition_bindings', {})
        bound_names = {obj['name'] for fields in bindings.values()
                       for objects in fields.values() for obj in objects}
        source_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in ('episode.json', 'task_scene.json', 'diagnostics.jsonl')
                         if (directory / name).is_file()}
        scene = m.get('scene', {})
        family = scene.get('pipeline')
        if scene.get('scene_file'):
            from .predictor_judge_observer import base_task_identity
            family = base_task_identity(scene).split('/')[0]
        top_z = _evidence_value(reason='world_surface_top_not_recorded')
        if 'diagnostics.jsonl' in source_hashes:
            surfaces = [json.loads(line).get('surface_info')
                        for line in (directory / 'diagnostics.jsonl').read_text().splitlines() if line.strip()]
            surfaces = [s for s in surfaces if isinstance(s, dict)]
            # finalize_base._fresh_surface_info records the same world top_z
            # for both frames; world_usable_rect only replaces the XY bounds
            # with an inscribed placement rectangle on round support surfaces.
            if surfaces and all(s.get('frame') in ('world_aabb', 'world_usable_rect') and
                    type(s.get('top_z')) in (int, float) and np.isfinite(s['top_z']) for s in surfaces):
                heights = {s['top_z'] for s in surfaces}
                top_z = (_evidence_value(heights.pop()) if len(heights) == 1 else
                         _evidence_value(reason='conflicting_surface_heights'))
        result = dict(episode_id=m['episode_id'], family=family, source_observations=n,
            source_transitions=m['transitions'], bindings=bindings,
            source_spec=m.get('scene', {}).get('ltl_safety', {}),
            task_instruction=m.get('scene', {}).get('prompt', ''),
            target_name=m.get('scene', {}).get('target_name'),
            unresolved_bindings=sorted(bound_names - set(names)), source_hashes=source_hashes,
            support_geometry=dict(
                diagnostics_reference='diagnostics.jsonl' if 'diagnostics.jsonl' in source_hashes else None,
                scene_reference='task_scene.json' if 'task_scene.json' in source_hashes else None,
                surface_name=m.get('scene', {}).get('surface_name'), top_z=top_z),
            observations=[])
        for step in ids:
            row = reader.read_observation(step)
            poses = row.get('objects', {}).get('pose_world')
            if poses is not None and np.asarray(poses).shape != (len(names), 7):
                raise ValueError('Object pose/name layout mismatch')
            objects = {}
            for i, name in enumerate(names):
                details = row.get('object_details', {}).get(name, {})
                obj = dict(pose_world=_evidence_value(poses[i]) if poses is not None else
                           _evidence_value(reason='pose_not_recorded'))
                for key in ('linear_velocity', 'angular_velocity', 'contacts'):
                    obj[key] = _source_measurement(details.get(key))
                obj['joints'] = {key: _source_measurement(value)
                                 for key, value in details.get('joints', {}).items()}
                objects[name] = obj
            safety = row.get('safety', {})
            ap_names = m.get('ap_names', [])
            values, valid = safety.get('ap_values', []), safety.get('ap_valid', [])
            if len(values) != len(ap_names) or len(valid) != len(ap_names):
                raise ValueError('Source AP value/validity layout mismatch')
            aps = {name: _evidence_value(bool(values[i])) if valid[i] else
                   _evidence_value(reason='invalid_source_ap') for i, name in enumerate(ap_names)}
            snapshot = None
            if m.get('snapshots', 0):
                try:
                    snapshot = reader.read_snapshot(step)
                except KeyError:
                    pass  # A sparse checkpoint is not evidence of "not grasped".
            grasp = _evidence_value(reason='snapshot_not_recorded')
            liquid = dict(initial_counts=None, system_particle_counts={}, snapshot_reference=None,
                          contained_counts=_evidence_value(reason='snapshot_not_recorded'),
                          particle_coordinate_frame='scene')
            if snapshot is not None:
                state, checkpoint = snapshot
                registry = state.get('registry', {})
                grasp_states = {name: value['ag_obj_constraint_params']
                                for name, value in registry.get('object_registry', {}).items()
                                if 'ag_obj_constraint_params' in value}
                grasp = _evidence_value(grasp_states) if grasp_states else _evidence_value(
                    reason='assisted_grasp_state_not_recorded')
                systems = registry.get('system_registry', {})
                liquid['system_particle_counts'] = {
                    name: sum(int(p['n_particles']) for p in value['particle_states'].values())
                    for name, value in systems.items() if 'particle_states' in value}
                memory = ((checkpoint or {}).get('monitor') or {}).get('predicate_memory', {})
                initial = memory.get('liquid_spilled', {}).get('initial_counts')
                liquid['initial_counts'] = _evidence_value(initial)['value']
                liquid['snapshot_reference'] = dict(path='snapshots.hdf5', observation_id=step)
                liquid['contained_counts'] = _evidence_value(
                    reason='offline_containment_geometry_not_resolved')
                if liquid_volumes is not None and 'liquid_spilled' in bindings:
                    subjects = bindings['liquid_spilled'].get('over', [])
                    liquid['geometry_sha256'] = {o['name']:liquid_volumes[o['name']].sha256
                        for o in subjects if o['name'] in liquid_volumes}
                    if not subjects or any(o['name'] not in liquid_volumes for o in subjects):
                        liquid['contained_counts'] = _evidence_value(reason='container_geometry_not_available')
                    else:
                        try:
                            definition = result['source_spec'].get('propositions', {}).get('liquid_spilled', {})
                            system = systems[definition.get('system_name', 'water')]
                            positions = []
                            for instancer in system['particle_states'].values():
                                points = np.asarray(instancer['particle_positions']).reshape(-1, 3)
                                if len(points) != instancer['n_particles']:
                                    raise ValueError('Particle position/count mismatch')
                                positions.append(points)
                            points = np.concatenate(positions) if positions else np.empty((0, 3))
                            scene_pose = np.r_[state['pos'], state['ori']]
                            counts = {}
                            for obj in subjects:
                                name = obj['name']; pose = objects[name]['pose_world']
                                if not pose['available']:
                                    raise ValueError('Container pose unavailable')
                                counts[name] = liquid_volumes[name].count(points,
                                    scene_pose=scene_pose, body_pose=pose['value'])
                            liquid['contained_counts'] = _evidence_value(counts)
                        except (KeyError, ValueError, TypeError) as exc:
                            liquid['contained_counts'] = _evidence_value(
                                reason='invalid_liquid_geometry_evidence: '+str(exc))
            result['observations'].append(dict(step=step, sim_time_s=float(row['sim_time_s']),
                physics_tick=int(row['physics_tick']), objects=objects,
                robot={key: _evidence_value(value) for key, value in row.get('robot', {}).items()},
                source_ap=aps, source_monitor_valid=bool(safety.get('monitor_valid', False)),
                assisted_grasp=grasp, liquid=liquid))
        result['evidence_sha256'] = hashlib.sha256(json.dumps(
            result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return result
