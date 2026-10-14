from pathlib import Path
import pytest
import os
from safetyjev.contact_reports import ContactPairs
from safetyjev import maniguard


def event(cache, kind='CONTACT_PERSIST', **kw):
    args=dict(stage_id=1,kind=kind,actor0=10,actor1=20,collider0=11,collider1=21,count=2)
    args.update(kw);cache.update(**args)


def test_sleep_does_not_remove_observed_contacts_but_lost_does():
    c=ContactPairs(1);event(c)
    assert c.neighbors({10})=={20} and c.neighbors({20})=={10}
    # Sleeping produces no event. Never infer lost contact from missing updates.
    assert c.neighbors({10})=={20}
    event(c,'CONTACT_LOST',count=0)
    assert not c.neighbors({10})


def test_multiple_shapes_loss_of_one_does_not_remove_other():
    c=ContactPairs(1);event(c);event(c,collider0=12)
    event(c,'CONTACT_LOST',actor0=20,actor1=10,collider0=21,collider1=11,count=0)
    assert c.neighbors({10})=={20}
    event(c,'CONTACT_LOST',collider0=12,count=0)
    assert not c.neighbors({10})


def test_reset_removal_empty_and_unknown_events():
    c=ContactPairs(1)
    assert not c.neighbors({10})
    event(c,count=0);assert not c.neighbors({10})
    event(c);c.remove({20});assert not c.pairs
    event(c);c.clear();assert not c.pairs
    with pytest.raises(ValueError):event(c,stage_id=2)
    with pytest.raises(ValueError):event(c,kind='UNKNOWN')


def test_opt_in_instrumentation_keeps_initial_validation_and_warmup():
    root=os.environ.get('SAFETYJEV_MANIGUARD_ROOT')
    if not root:pytest.skip('Pinned ManiGuard source required')
    source=(Path(root)/'maniguard/eval/benchmark.py').read_text()
    plain=maniguard.instrument(source,'codex_harness')
    fixed=maniguard.instrument(source,'codex_harness',persistent_contacts=True)
    assert 'prepare_contact_reports(env)' not in plain
    assert fixed.index('prepare_contact_reports(env)')<fixed.index('initialization_phase = "warmup"')<fixed.index('_validate_initial_safety(monitor, scene_info["name"])')
    assert fixed.count('for _ in range(10):')==source.count('for _ in range(10):')
    assert 'close_contact_reports()' in fixed


def test_adapter_preserves_force_queries_ignores_removed_objects_and_fails_on_bad_stream(monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from safetyjev.contact_reports import _ContactAdapter
    class Link:
        contact_reporting_enabled=True
        def __init__(self, path): self.prim_path=path
    class Obj:
        kinematic_only=False
        def __init__(self,name,path):self.name=name;self.links={'root':Link(path)};self.wakes=0
        def wake(self):self.wakes+=1
    a,b=Obj('jar','10'),Obj('table','20');b.kinematic_only=True
    class State:
        def __init__(self,obj):self.obj=obj
        def _get_value(self,ignore_objs=None,non_zero_impulse=False):return set()
    callbacks={};receiver=[]
    def subscribe(f):receiver.append(f);return NS()
    sim=NS(stage_id=1,psi=NS(subscribe_contact_report_events=subscribe),
           add_callback_on_stop=lambda n,f:callbacks.update(stop=f),
           add_callback_on_remove_obj=lambda n,f:callbacks.update(remove=f),
           remove_callback_on_stop=lambda n:callbacks.pop('stop'),
           remove_callback_on_remove_obj=lambda n:callbacks.pop('remove'))
    lazy=NS(pxr=NS(PhysicsSchemaTools=NS(sdfPathToInt=int)))
    monkeypatch.setitem(sys.modules,'omnigibson',NS(lazy=lazy))
    original=State._get_value
    adapter=_ContactAdapter(NS(sim=sim),NS(scene=NS(objects=[a,b])),State)
    try:
        receiver[0]([NS(stage_id=1,type='ContactEventType.CONTACT_PERSIST',actor0=10,actor1=20,collider0=11,collider1=21,num_contact_data=1)],[])
        assert State(a)._get_value()=={b.links['root']}
        assert not State(a)._get_value(non_zero_impulse=True)
        assert not State(a)._get_value(ignore_objs=(b,))
        assert a.wakes==1 and b.wakes==0
        callbacks['remove'](b)
        assert not State(a)._get_value()
        receiver[0]([NS(stage_id=99,type='ContactEventType.CONTACT_PERSIST',actor0=10,actor1=20,collider0=11,collider1=21,num_contact_data=1)],[])
        with pytest.raises(RuntimeError,match='Invalid contact event stream'):State(a)._get_value()
    finally:adapter.close()
    assert State._get_value is original and not callbacks

    def fail_wake(): raise RuntimeError('wake failed')
    a.wake=fail_wake
    with pytest.raises(RuntimeError,match='wake failed'):
        _ContactAdapter(NS(sim=sim),NS(scene=NS(objects=[a,b])),State)
    assert State._get_value is original and not callbacks
