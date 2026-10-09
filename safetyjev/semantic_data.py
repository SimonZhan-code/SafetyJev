"""Semantic annotation storage and temporal labels, independent of model inputs."""
import hashlib
import json
from pathlib import Path

from .semantic_labels import liquid_interval_label


def _digest(annotation):
    content = {key: value for key, value in annotation.items() if key != 'annotation_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_annotation(path, annotation):
    content = dict(annotation, annotation_sha256=_digest(annotation))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def read_annotation(path):
    annotation = json.loads(Path(path).read_text(encoding='utf-8'))
    if annotation.get('annotation_sha256') != _digest(annotation):
        raise ValueError('Semantic annotation digest mismatch')
    return annotation


def prepare_semantic_annotation(directory, episode_id, definitions, *, review=False, reuse_annotations=None,
                                liquid_asset_root=None, liquid_resolver=None):
    """Derive once or reuse a verified annotation of an immutable source episode.

    Large HDF5 files use size/mtime invalidation, not a claim of a full content
    checksum. Small metadata and extraction code are hashed. Raw files must stay
    immutable; copied sources with changed timestamps must be prepared again.
    """
    from . import source_episodes, semantic_labels, liquid_geometry

    directory = Path(directory)
    def identity():
        files = {}
        for name in ('episode.json', 'task_scene.json', 'diagnostics.jsonl',
                     'trajectory.hdf5', 'snapshots.hdf5'):
            path = directory/name
            if not path.is_file():
                files[name] = None
            elif path.suffix == '.hdf5':
                stat = path.stat()
                files[name] = dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            else:
                files[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return files

    source_files = identity()
    implementation = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                      for m in (source_episodes, semantic_labels, liquid_geometry)}
    implementation[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if reuse_annotations is not None:
        name = hashlib.sha256(episode_id.encode()).hexdigest()+'.json'
        annotation = read_annotation(Path(reuse_annotations)/name)
        if annotation.get('source_files') != source_files:
            raise ValueError('Source files changed or annotation lacks reuse provenance')
        if annotation.get('annotation_implementation') != implementation:
            raise ValueError('Source annotation implementation changed; rebuild annotations')
        # Definition/status/episode identity are also checked by the sample builder.
        return annotation
    volumes, assets = None, {}
    if any(d['label_rule']['kind'] == 'liquid_net_loss' for d in definitions['definitions']):
        metadata = json.loads((directory/'episode.json').read_text())
        subjects = metadata.get('proposition_bindings', {}).get('liquid_spilled', {}).get('over', [])
        if subjects:
            if liquid_resolver is None:
                import os
                root = liquid_asset_root or os.environ.get('OMNIGIBSON_DATA_PATH')
                if not root:
                    raise ValueError('Liquid preparation requires --liquid-asset-root or OMNIGIBSON_DATA_PATH')
                liquid_resolver = liquid_geometry.LiquidAssetResolver(root)
            scene = json.loads((directory/'task_scene.json').read_text())
            volumes, assets = liquid_resolver.resolve(scene, [o['name'] for o in subjects])
    annotation = semantic_labels.annotate_semantic_episode(
        source_episodes.extract_semantic_evidence(directory, liquid_volumes=volumes), definitions, review=review)
    annotation['liquid_assets'] = assets
    if identity() != source_files:
        raise ValueError('Source files changed during annotation preparation')
    annotation.update(source_files=source_files, annotation_implementation=implementation)
    return annotation


def evaluate_semantic_interval(annotation, query_id, start_step, end_step, *, task):
    """Evaluate a scope-specific query, not an ambiguous shared semantic ID.

    Call only on actions verified as executed by the sample builder. This pure
    function deals with supervision time, not execution provenance. Classifier
    is current-frame-only; interval questions are explicitly input-ineligible.
    """
    if (task not in ('classifier', 'predictor_judge') or type(start_step) is not int
            or type(end_step) is not int or start_step < 0 or end_step < start_step
            or (task == 'predictor_judge' and start_step == end_step)):
        raise ValueError('Invalid semantic task or interval')
    if task == 'classifier' and start_step != end_step:
        raise ValueError('Current classifier requires a single observation')
    query = annotation['queries'][query_id]
    states = {s['step']: s for s in query['states']}
    if len(states) != len(query['states']):
        raise ValueError('Duplicate semantic observation indices')
    return _evaluate_interval(query, states, start_step, end_step, task=task,
                              reset_steps=annotation.get('reset_steps', []))


def _evaluate_interval(query, states, start_step, end_step, *, task,
                       reset_steps=(), available_end=None, boundary_reason='censored'):
    """Lookup tables are shared across windows so episode construction stays linear."""
    current = states.get(start_step, {}).get('label')
    result = dict(label=None, reason=None, evidence_steps=[],
                  starts_violated=None if current is None else bool(current))
    if query['temporal_kind'] == 'interval':
        if task == 'classifier':
            result['reason'] = 'input_history_required'
            return result
        expected = list(range(start_step, end_step+1))
        if any(start_step < t <= end_step for t in reset_steps):
            result['reason'] = 'reset_in_interval'
            return result
        if available_end is not None and end_step > available_end:
            result['reason'] = boundary_reason
            return result
        if any(t not in states for t in expected):
            result['reason'] = 'censored'
            return result
        rule = query['label_rule']
        if 'loss_fraction' not in rule or 'minimum_initial_count' not in rule:
            result['reason'] = 'liquid_definition_not_selected'
            return result
        by_object = {}
        for obj in query['objects']:
            counts = []
            for t in expected:
                record = states[t].get('contained_counts', {})
                values = record.get('value')
                counts.append(values.get(obj['name']) if record.get('available') and isinstance(values, dict) else None)
            value = liquid_interval_label(counts, 0, len(counts)-1,
                threshold=rule['loss_fraction'], minimum_initial_count=rule['minimum_initial_count'],
                reset_steps=[t-start_step for t in reset_steps if start_step < t <= end_step])
            value['evidence_steps'] = expected
            by_object[obj['name']] = value
        labels = [v['label'] for v in by_object.values()]
        label = 1 if 1 in labels else (0 if labels and all(v == 0 for v in labels) else None)
        result.update(label=label, reason='evaluated' if label is not None else 'liquid_evidence_unavailable',
                      evidence_steps=expected, by_object=by_object)
        return result
    if query['temporal_kind'] not in ('state', 'event'):
        raise ValueError('Unknown semantic temporal kind')
    if task == 'classifier':
        row = states.get(start_step)
        result.update(label=current, reason=row['reason'] if row else 'observation_missing',
                      evidence_steps=[start_step] if row else [])
        return result
    for t in range(start_step+1, end_step+1):
        if t in reset_steps:
            result['reason'] = 'reset_in_interval'
            return result
        if available_end is not None and t > available_end:
            result['reason'] = boundary_reason
            return result
        row = states.get(t)
        if row is None:
            result['reason'] = 'censored'
            return result
        result['evidence_steps'].append(t)
        if row['label'] is None:
            result['reason'] = row['reason']
            return result
        if row['label'] == 1:
            result.update(label=1, reason='observed_positive')
            return result
    result.update(label=0, reason='complete_negative')
    return result


def build_semantic_samples(record, annotation, definitions, *, task,
                           history_frames=3, review=False, input_contract='per_step_v1'):
    """Build indices only: current classification or committed-suffix judgment.

    Source media and actions stay in the record. Neither physical label evidence
    nor the legacy absorbing monitor state becomes part of model inputs.
    """
    import numpy as np
    from .semantic_labels import validate_semantic_definitions
    from .predictor_judge_schema import validate_episode_record

    validate_semantic_definitions(definitions, for_production=not review)
    validate_episode_record(record)
    if task not in ('classifier', 'predictor_judge'):
        raise ValueError('Unsupported semantic task')
    if input_contract not in ('per_step_v1','chunk_start_v2'):
        raise ValueError('Unsupported input contract')
    chunk_mode=task=='predictor_judge' and input_contract=='chunk_start_v2'
    if chunk_mode:history_frames=8
    if type(history_frames) is not int or history_frames < 1:
        raise ValueError('Positive adjacent history length required')
    if annotation['episode_id'] != record['episode_id']:
        raise ValueError('Annotation and source episode disagree')
    definition_hash = hashlib.sha256(json.dumps(definitions, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if annotation['definition_sha256'] != definition_hash:
        raise ValueError('Annotation definition hash mismatch')
    if not review and annotation['status'] != 'reviewed':
        raise ValueError('Annotation requires review')
    if 'annotation_sha256' in annotation and annotation['annotation_sha256'] != _digest(annotation):
        raise ValueError('Annotation digest mismatch')
    lookup = {}
    for qid, query in annotation['queries'].items():
        states = {s['step']: s for s in query['states']}
        if len(states) != len(query['states']) or any(type(t) is not int or t < 0 for t in states):
            raise ValueError('Invalid semantic observation indices')
        if any(type(s['label']) is not int or s['label'] not in (0,1)
               for s in states.values() if s['label'] is not None):
            raise ValueError('Semantic labels must be binary or unavailable')
        lookup[qid] = states
    annotation_hash = _digest(annotation)
    proposals = {p['proposal_id']: p for p in record['proposals']}
    execution = {e['step']: e for e in record['execution']}
    observations = record['observations']
    steps = range(len(observations)) if task == 'classifier' else sorted(execution)
    resets = set(annotation.get('reset_steps', []))
    rows = []
    for t in steps:
        if chunk_mode and execution[t]['offset']!=0:continue
        end = observed_end = t
        boundary_reason = 'censored'
        input_valid = True
        action_fields = {}
        if task == 'predictor_judge':
            action = execution[t]; proposal = proposals[action['proposal_id']]
            offset = action['offset']
            commands = np.asarray(proposal['planned_commands'], dtype=np.float32)[offset:]
            end = t + len(commands)
            input_valid = bool(np.isfinite(commands).all())
            for step in range(t, end):
                actual = execution.get(step)
                if actual is None or step+1 >= len(observations):
                    break
                if (actual['proposal_id'] != proposal['proposal_id'] or actual['offset'] != offset+step-t
                        or not np.array_equal(np.asarray(actual['command'], dtype=np.float32), commands[step-t])):
                    boundary_reason = 'action_replaced'
                    break
                observed_end = step+1
            action_fields = dict(proposal_id=proposal['proposal_id'], action_offset=offset,
                                 valid_steps=len(commands))
        history = [t] if task == 'classifier' else [s if s >= 0 else None for s in range(t-history_frames+1,t+1)]
        if chunk_mode:
            # o_(i+1) is paired with actually executed a_i. o_0 has no action.
            boundary=max([0]+[r for r in resets if r<=t])
            if t>boundary:
                previous=execution[t-1]['proposal_id']
                boundary=max(boundary,t-8)
                for s in range(t-1,boundary-1,-1):
                    if execution[s]['proposal_id']!=previous:
                        boundary=s+1;break
            past=[s if s>=boundary else None for s in range(t-8,t)]
            history=[s+1 if s is not None else None for s in past]
            if all(s is None for s in history):history[-1]=t
            action_fields.update(input_contract=input_contract,history_action_steps=past)
        for qid, query in annotation['queries'].items():
            verdict = _evaluate_interval(query, lookup[qid], t, end, task=task,
                reset_steps=resets, available_end=observed_end, boundary_reason=boundary_reason)
            if any(not observations[s].get('valid', True) for s in history if s is not None):
                verdict.update(label=None, reason='invalid_observation')
            if any(s is not None and s < reset <= t for s in history for reset in resets):
                verdict.update(label=None, reason='reset_in_history')
            if not input_valid:
                verdict.update(label=None, reason='invalid_actions')
            if chunk_mode and len(commands)!=8:
                verdict.update(label=None, reason='chunk_length_not_eight')
            label = verdict['label']
            rows.append(dict(id=f"{record['episode_id']}:{task}:{t}:{qid}",
                episode_id=record['episode_id'], group_id=record['group_id'], task=task,
                constraint_id=qid, semantic_id=query['semantic_id'],
                question=(('During execution of the remaining planned actions: ' if task == 'predictor_judge' else '')
                          + query['question']),
                requires_history=query['temporal_kind']=='interval',
                constraint_context={'source':'none','text':''},
                start_step=t, end_step=end, history_steps=history, label_interval=[t,end],
                target=None if label is None else [float(1-label),float(label)],
                label_reason=verdict['reason'], starts_violated=verdict['starts_violated'],
                observed_end=observed_end, definition_sha256=definition_hash,
                annotation_sha256=annotation_hash, **action_fields))
    return rows


def to_classifier_record(sample, record):
    """Use the existing Noul envelope with explicit source-image references."""
    if sample['task'] != 'classifier':
        raise ValueError('Expected current-state classifier sample')
    t=sample['start_step']
    metadata={k:v for k,v in sample.items() if k not in ('question','target','constraint_context')}
    metadata['query_id']=sample['constraint_id']
    return dict(id=sample['id'],group_id=sample['group_id'],split=sample['split'],
        source='maniguard_source',question=sample['question'],kind='noul',options=['no','yes'],
        target=sample['target'],state={'observation_window':{
            'resource_id':record['episode_id'],'frame_indices':[t],
            'image_refs':{c:[ref] for c,ref in record['observations'][t]['images'].items()}}},
        metadata=metadata)
