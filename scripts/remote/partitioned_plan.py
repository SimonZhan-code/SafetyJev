"""Validate immutable ownership for the two-node ManiGuard evaluation."""
import hashlib
import json


def case_key(family, scene):
    return family + '/' + scene


def validate_assignment(assignment, resources):
    expected = {case_key(f, s) for f, spec in resources['families'].items()
                for scenes in spec['scenes_by_level'].values() for s in scenes}
    owners = {}
    for worker, spec in assignment['workers'].items():
        for key in spec['cases']:
            if key in owners:
                raise ValueError('Overlapping assignment: ' + key)
            owners[key] = worker
    if set(owners) != expected:
        raise ValueError('Assignment does not cover the exact benchmark case set')
    for worker, spec in assignment['workers'].items():
        queued = set()
        for phase in spec['phases']:
            for family in phase['families']:
                for scene in resources['families'][family]['scenes_by_level'][phase['level']]:
                    key = case_key(family, scene)
                    if owners[key] != worker or key in queued:
                        raise ValueError('Invalid or repeated queued case: ' + key)
                    queued.add(key)
        if set(spec['cases']) != queued | set(spec.get('existing_cases', [])):
            raise ValueError('Cases missing from worker execution plan: ' + worker)
        if queued & set(spec.get('existing_cases', [])):
            raise ValueError('Queue repeats existing base run')
    return owners


def authorize_selection(assignment, resources, worker, selected):
    owners = validate_assignment(assignment, resources)
    for family, spec in selected.items():
        for scene in spec['scenes']:
            key = case_key(family, scene)
            if owners[key] != worker:
                raise ValueError('Worker does not own ' + key)
    return hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
