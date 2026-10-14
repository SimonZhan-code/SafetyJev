"""Opt-in, pinned Isaac 5.1 contact persistence compatibility repair.

The native raw-body API drops sleeping contacts. Keep *observed* PhysX collider
pairs until CONTACT_LOST, rather than inferring contact from geometry. Force
queries still use the native API; no cached impulse is represented as current.
"""
from pathlib import Path
import hashlib
import inspect

PINNED_SOURCES = {
    'prims/rigid_prim.py': '44060e7bfdd88721624003ee3ad0de49fa1a008867637957744176e138068647',
    'prims/entity_prim.py': '4fa84d85199121e8335e6b49f95fe9fb93bf277c5ec603680fffde9a69cfdb10',
    'object_states/contact_bodies.py': 'd2046221be5900c9807c5ef0e3d052a48a0a3d4e0d8602041be56e85d248fec9',
}


class ContactPairs:
    def __init__(self, stage_id):
        self.stage_id = stage_id
        self.pairs = {}

    def update(self, stage_id, kind, actor0, actor1, collider0, collider1, count):
        if stage_id != self.stage_id:
            raise ValueError('Contact event belongs to a different stage')
        key = tuple(sorted(((actor0, collider0), (actor1, collider1))))
        if kind == 'CONTACT_LOST':
            self.pairs.pop(key, None)
        elif kind in ('CONTACT_FOUND', 'CONTACT_PERSIST'):
            if count > 0:
                self.pairs[key] = (actor0, actor1)
            else:
                self.pairs.pop(key, None)
        else:
            raise ValueError('Unsupported contact event: ' + kind)

    def neighbors(self, actors):
        others = set()
        for a, b in self.pairs.values():
            if a in actors: others.add(b)
            if b in actors: others.add(a)
        return others - set(actors)

    def remove(self, actors):
        self.pairs = {k: v for k, v in self.pairs.items() if not set(v).intersection(actors)}

    def clear(self):
        self.pairs.clear()


_active = None


def close_contact_reports():
    global _active
    if _active is not None:
        _active.close()
        _active = None


def prepare_contact_reports(env):
    """Call after reset, before existing hold warmup, never after policy actions."""
    global _active
    close_contact_reports()
    import omnigibson as og
    from omnigibson.object_states import ContactBodies
    root = Path(inspect.getfile(ContactBodies)).resolve().parents[1]
    for relative, expected in PINNED_SOURCES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Contact compatibility source mismatch: ' + relative)
    adapter = _ContactAdapter(og, env, ContactBodies)
    _active = adapter
    return adapter.provenance


class _ContactAdapter:
    def __init__(self, og, env, state_type):
        from omnigibson import lazy
        self.og, self.env, self.state_type = og, env, state_type
        self.cache = ContactPairs(og.sim.stage_id)
        self.error = None
        self.original = state_type._get_value
        self.links = {}
        self.object_actors = {}
        for obj in env.scene.objects:
            ids = set()
            for link in obj.links.values():
                actor = int(lazy.pxr.PhysicsSchemaTools.sdfPathToInt(link.prim_path))
                self.links[actor] = link
                ids.add(actor)
            self.object_actors[obj] = ids
        def receive(headers, data):
            try:
                for h in headers:
                    self.cache.update(h.stage_id, str(h.type).rsplit('.', 1)[-1],
                                      h.actor0, h.actor1, h.collider0, h.collider1, h.num_contact_data)
            except Exception as exc:
                self.error = str(exc)
        self.subscription = og.sim.psi.subscribe_contact_report_events(receive)
        self.callback_name = 'safetyjev_persistent_contacts'
        og.sim.add_callback_on_stop(self.callback_name, self.cache.clear)
        og.sim.add_callback_on_remove_obj(self.callback_name, self.remove_object)
        def get_value(state, ignore_objs=None, non_zero_impulse=False):
            if self.error is not None:
                raise RuntimeError('Invalid contact event stream: ' + self.error)
            current = self.original(state, ignore_objs, non_zero_impulse)
            if non_zero_impulse:
                return current
            actors = self.object_actors.get(state.obj)
            if actors is None:
                return current
            ignored = set().union(*(self.object_actors.get(o, set()) for o in (ignore_objs or ())))
            return current | {self.links[a] for a in self.cache.neighbors(actors) - ignored if a in self.links}
        self.get_value = get_value
        state_type._get_value = get_value
        # Restored sleeping bodies otherwise never emit their initial contact pairs.
        # Existing benchmark hold warmup supplies physics updates; no extra steps.
        awakened = []
        try:
            for obj in env.scene.objects:
                if not obj.kinematic_only and any(link.contact_reporting_enabled for link in obj.links.values()):
                    obj.wake()
                    awakened.append(obj.name)
        except Exception:
            self.close()
            raise
        self.provenance = {'mode': 'physx_pair_persistence_v1', 'source_sha256': dict(PINNED_SOURCES),
                           'awakened_objects': awakened, 'extra_physics_steps': 0,
                           'contact_basis': 'PhysX FOUND/PERSIST with contact points, removed on LOST; reset/stop/removal clears',
                           'sleep_thresholds_changed': False, 'predicate_definitions_changed': False}

    def remove_object(self, obj):
        actors = self.object_actors.pop(obj, set())
        self.cache.remove(actors)
        for actor in actors: self.links.pop(actor, None)

    def close(self):
        if self.state_type._get_value is self.get_value:
            self.state_type._get_value = self.original
        self.subscription = None
        self.cache.clear()
        self.og.sim.remove_callback_on_stop(self.callback_name)
        self.og.sim.remove_callback_on_remove_obj(self.callback_name)
