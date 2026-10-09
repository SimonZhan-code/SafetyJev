"""Contracts for derived semantic safety supervision, separate from source GT."""
import math
import re
from string import Formatter


_RULES = {
    'tilt_angle': ('state', 'object_pose'),
    'below_surface': ('state', 'object_height_and_support_plane'),
    'liquid_net_loss': ('interval', 'contained_particle_counts'),
    'direct_contact': ('state', 'contact_evidence'),
    'ap_all_of': ('state', 'source_ap_states'),
}
_FAMILIES = {'jar_transport', 'lid_transport', 'stack_retrieve',
             'cabinet_pickup', 'dusty_transfer', 'clutter_pickup'}


def _number(value, low, high, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low < value <= high):
        raise ValueError('Invalid ' + name)


def validate_semantic_definitions(definitions, *, for_production=False):
    """Validate reviewed rules or inspect candidates without activating them.

    Call with for_production=True at production build boundaries. Merely being
    syntactically valid does not approve a proposed labeling threshold.
    """
    if not isinstance(definitions, dict) or definitions.get('schema_version') != 1:
        raise ValueError('Unsupported semantic schema')
    if definitions.get('yes_means') != 'violation':
        raise ValueError('Yes must mean violation')
    rows = definitions.get('definitions')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing semantic definitions')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid semantic definition')
        sid = row.get('semantic_id')
        if not isinstance(sid, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', sid) or sid in seen:
            raise ValueError('Duplicate or invalid semantic ID')
        seen.add(sid)
        status = row.get('definition_status')
        if status not in ('review_only', 'reviewed') or (for_production and status != 'reviewed'):
            raise ValueError('Semantic definition requires review: ' + sid)
        question = row.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Missing semantic question')
        for _, field, spec, conversion in Formatter().parse(question):
            if field is not None and (field != 'subject' or spec or conversion):
                raise ValueError('Question permits only the subject placeholder')
        bindings = row.get('scope_bindings')
        if not isinstance(bindings, dict) or not bindings or not set(bindings) <= _FAMILIES:
            raise ValueError('Missing or unsupported family scope')
        for names in bindings.values():
            if (not isinstance(names, list) or not names or
                    any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names)):
                raise ValueError('Scope must name source propositions')
        rule = row.get('label_rule')
        if not isinstance(rule, dict) or rule.get('kind') not in _RULES:
            raise ValueError('Unsupported semantic label rule')
        temporal, evidence = _RULES[rule['kind']]
        if row.get('temporal_kind') != temporal:
            raise ValueError('Rule and temporal kind disagree')
        if rule.get('evidence') != evidence:
            raise ValueError('Rule requires physical evidence: ' + evidence)
        if rule['kind'] in ('tilt_angle', 'liquid_net_loss', 'below_surface'):
            liquid = rule['kind'] == 'liquid_net_loss'
            candidate_key, selected_key, upper = {
                'liquid_net_loss': ('candidate_loss_fractions', 'loss_fraction', 1),
                'tilt_angle': ('candidate_thresholds_deg', 'threshold_deg', 180),
                'below_surface': ('candidate_margins_m', 'margin_m', math.inf),
            }[rule['kind']]
            candidates = rule.get(candidate_key)
            if not isinstance(candidates, list) or not candidates:
                raise ValueError('Missing candidate thresholds')
            for value in candidates:
                _number(value, 0, upper, 'candidate threshold')
            if selected_key in rule:
                _number(rule[selected_key], 0, upper, 'selected threshold')
            elif status == 'reviewed':
                raise ValueError('Reviewed rule requires a selected threshold')
            if liquid and status == 'reviewed':
                minimum = rule.get('minimum_initial_count')
                if type(minimum) is not int or minimum < 1:
                    raise ValueError('Reviewed liquid rule requires a minimum initial count')
        if rule['kind'] == 'ap_all_of':
            terms = rule.get('all_of')
            if (not isinstance(terms, dict) or not terms or
                    any(type(v) is not bool for v in terms.values()) or
                    any(set(terms) != set(names) for names in bindings.values())):
                raise ValueError('AP conjunction must match its declared scope')


def tilt_label(quaternion_xyzw, *, threshold_deg):
    """Local +Z against world +Z, matching the source Upright axis convention."""
    _number(threshold_deg, 0, 180, 'tilt threshold')
    try:
        x, y, z, w = map(float, quaternion_xyzw)
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError
    except (TypeError, ValueError):
        return dict(label=None, reason='invalid_orientation', angle_deg=None)
    x, y, z, w = (v/norm for v in (x,y,z,w))
    up_x, up_y, up_z = 2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)
    angle = math.degrees(math.atan2(math.hypot(up_x, up_y), up_z))
    # Resolve floating-point roundoff at the mathematical boundary, not sensor noise.
    exceeds = angle > threshold_deg and not math.isclose(angle, threshold_deg, rel_tol=0, abs_tol=1e-10)
    return dict(label=int(exceeds), reason='evaluated', angle_deg=angle)


def below_surface_label(object_z, surface_top_z, *, margin_m):
    """Current object origin below the recorded world tabletop plane by a margin.

    This geometric proxy does not establish an uncontrolled fall or persist after
    recovery. It deliberately does not depend on the floor or contact history.
    """
    _number(margin_m, 0, math.inf, 'surface margin')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (object_z, surface_top_z)):
        return dict(label=None, reason='height_evidence_unavailable', height_below_surface_m=None)
    depth = surface_top_z - object_z
    exceeds = depth > margin_m and not math.isclose(depth, margin_m, rel_tol=0, abs_tol=1e-10)
    return dict(label=int(exceeds), reason='evaluated', height_below_surface_m=depth,
                object_z_m=object_z, surface_top_z_m=surface_top_z)


def liquid_interval_label(counts, start_step, end_step, *, threshold,
                          minimum_initial_count, reset_steps=()):
    """Endpoint net loss. Peak loss is diagnostic and cannot label a truncated tail."""
    from fractions import Fraction
    from numbers import Integral
    _number(threshold, 0, 1, 'liquid threshold')
    if type(minimum_initial_count) is not int or minimum_initial_count < 1:
        raise ValueError('Positive minimum initial count required')
    if type(start_step) is not int or type(end_step) is not int or not 0 <= start_step < end_step:
        raise ValueError('Invalid liquid interval')
    result = dict(label=None, reason=None, loss_fraction=None, peak_loss_fraction=None,
                  evidence_steps=list(range(start_step, min(end_step+1, len(counts)))))
    if end_step >= len(counts):
        result['reason'] = 'censored'
    elif any(start_step < s <= end_step for s in reset_steps):
        result['reason'] = 'reset_in_interval'
    else:
        values = counts[start_step:end_step+1]
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in values):
            result['reason'] = 'invalid_liquid_count'
        elif values[0] < minimum_initial_count:
            result['reason'] = 'insufficient_initial_liquid'
        else:
            start, end = int(values[0]), int(values[-1])
            limit = Fraction(str(threshold))
            result.update(label=int((start-end)*limit.denominator >= start*limit.numerator),
                reason='evaluated', loss_fraction=(start-end)/start,
                peak_loss_fraction=(start-int(min(values)))/start)
    return result


def _subject(objects):
    categories = sorted({o.get('category', 'object').replace('_', ' ') for o in objects})
    if len(objects) == 1:
        return 'the ' + categories[0]
    if len(categories) == 1:
        return 'any ' + categories[0]
    return 'any ' + ', '.join(categories[:-1]) + ' or ' + categories[-1]


def _scoped_subject(objects, scope, evidence):
    """Disambiguate same-category scopes using static task roles, never labels.

    Resolved name sets matter: category-glob bindings can overlap, so an AP
    named 'obstacles' does not by itself authorize excluding the task target.
    """
    names = {obj['name'] for obj in objects}
    categories = {obj.get('category', 'object') for obj in objects}
    other_objects = [obj for binding in evidence['bindings'].values()
                     for obj in binding.get('over', []) if obj['name'] not in names]
    subject = _subject(objects)
    if not any(obj.get('category', 'object') in categories for obj in other_objects):
        return subject, ''
    instruction = evidence.get('task_instruction', '')
    target = evidence.get('target_name')
    if not isinstance(instruction, str) or not instruction.strip() or not target:
        raise ValueError('Ambiguous object scope requires a static task instruction and target')
    if names == {target}:
        subject += ' selected as the task target'
    else:
        # A role cannot distinguish an arbitrary subset of same-category
        # obstacles/stack members. Only the separately grounded target may be
        # omitted from a category-wide group without another visible descriptor.
        if any(obj['name'] != target and obj.get('category', 'object') in categories
               for obj in other_objects):
            raise ValueError('Ambiguous object scope excludes unresolved non-target objects: ' + scope)
        if scope in ('obstacles_upright', 'obstacles_dropped', 'obstacle_dropped') and target not in names:
            subject += ' among the task obstacles (excluding the task target)'
        elif scope in ('all_stack_upright', 'any_stack_dropped'):
            relation = 'including' if target in names else 'excluding'
            subject += f' in the stack ({relation} the task target)'
        else:
            raise ValueError('Ambiguous object scope cannot be grounded by the supported task roles: ' + scope)
    return subject, instruction.strip()


def _ap(row, name):
    value = row.get('source_ap', {}).get(name, {})
    return value.get('value') if value.get('available') and type(value.get('value')) is bool else None


def annotate_semantic_episode(evidence, definitions, *, review=False):
    """Derive review or approved annotations; never mutate the original monitor GT.

    Non-conjunction scope entries create separate questions, e.g. target versus
    obstacles. A source AP is reused only for unchanged contact/support/closure
    meaning, never to reconstruct a new angular or liquid threshold.
    """
    import hashlib
    import json
    validate_semantic_definitions(definitions, for_production=not review)
    family = evidence.get('family')
    if family not in _FAMILIES:
        raise ValueError('Missing or unsupported evidence family')
    rows = evidence['observations']
    steps = [r['step'] for r in rows]
    if not rows or steps != sorted(set(steps)):
        raise ValueError('Evidence steps must be unique and increasing')
    annotation = dict(schema_version=1, episode_id=evidence['episode_id'], family=family,
        evidence_sha256=evidence['evidence_sha256'], source_hashes=evidence['source_hashes'],
        definition_sha256=hashlib.sha256(json.dumps(definitions, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        status='review_only' if review else 'reviewed', queries={}, excluded=[])
    for definition in definitions['definitions']:
        ap_names = definition['scope_bindings'].get(family, [])
        if not ap_names:
            continue
        rule = definition['label_rule']; kind = rule['kind']; sid = definition['semantic_id']
        conjunction = kind == 'ap_all_of'
        groups = [ap_names] if conjunction else [[name] for name in ap_names]
        for group in groups:
            if not all(name in evidence['bindings'] for name in group):
                annotation['excluded'].append(dict(semantic_id=sid, source_aps=group, reason='scope_not_present'))
                continue
            # Conjunction's first AP refers to the carried object; lid AP may refer to the lid.
            objects = evidence['bindings'][group[0]].get('over', [])
            if not objects:
                annotation['excluded'].append(dict(semantic_id=sid, source_aps=group, reason='empty_object_scope'))
                continue
            if kind == 'tilt_angle' and 'threshold_deg' not in rule:
                raise ValueError('Select an explicit review tilt threshold')
            if kind == 'below_surface' and 'margin_m' not in rule:
                raise ValueError('Select an explicit review surface margin')
            qid = sid if conjunction else sid + ':' + group[0]
            subject, instruction = _scoped_subject(objects, group[0], evidence)
            query = dict(semantic_id=sid, temporal_kind=definition['temporal_kind'],
                source_aps=group, objects=objects, subject=subject, label_rule=dict(rule), states=[])
            query['question'] = definition['question'].format(subject=query['subject'])
            if instruction:
                query['question'] = 'Task for identifying the objects: ' + instruction + '\n' + query['question']
            for row in rows:
                value = dict(step=row['step'], sim_time_s=row['sim_time_s'], label=None, reason=None)
                if kind == 'tilt_angle':
                    angles = {}
                    for obj in objects:
                        pose = row.get('objects', {}).get(obj['name'], {}).get('pose_world', {})
                        quat = pose['value'][3:] if pose.get('available') and len(pose.get('value', [])) == 7 else None
                        angles[obj['name']] = tilt_label(quat, threshold_deg=rule['threshold_deg'])
                    labels = [a['label'] for a in angles.values()]
                    value.update(label=1 if 1 in labels else (None if None in labels else 0),
                        reason='evaluated' if 1 in labels or None not in labels else 'orientation_unavailable',
                        angles=angles, basis='object_orientation')
                elif kind == 'below_surface':
                    surface = evidence.get('support_geometry', {}).get('top_z', {})
                    top_z = surface.get('value') if surface.get('available') else None
                    heights = {}
                    for obj in objects:
                        pose = row.get('objects', {}).get(obj['name'], {}).get('pose_world', {})
                        z = pose['value'][2] if pose.get('available') and len(pose.get('value', [])) == 7 else None
                        heights[obj['name']] = below_surface_label(z, top_z, margin_m=rule['margin_m'])
                    labels = [h['label'] for h in heights.values()]
                    value.update(label=1 if 1 in labels else (None if None in labels else 0),
                        reason='evaluated' if 1 in labels or None not in labels else 'height_evidence_unavailable',
                        heights=heights, basis='height_below_support')
                elif conjunction:
                    terms = [_ap(row, name) for name in group]
                    truth = [None if v is None else v == rule['all_of'][name] for name,v in zip(group,terms)]
                    label = 0 if False in truth else (None if None in truth else 1)
                    value.update(label=label, reason='source_ap_unavailable' if label is None else 'evaluated',
                        basis='source_ap_conjunction')
                elif kind == 'direct_contact':
                    contact = _ap(row, group[0])
                    value.update(label=None if contact is None else int(contact), basis='source_contact_ap',
                        reason='source_ap_unavailable' if contact is None else 'evaluated')
                elif kind == 'liquid_net_loss':
                    value.update(reason='interval_evaluation_required',
                        contained_counts=row.get('liquid', {}).get('contained_counts',
                            dict(available=False, value=None, reason='containment_unavailable')))
                query['states'].append(value)
            annotation['queries'][qid] = query
    return annotation
