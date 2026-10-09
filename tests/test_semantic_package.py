"""Small real HDF5 packages exercise existing consumers without simulator startup."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from test_source_episodes import source_fixture
from safetyjev.predictor_judge_dataset import build_package, PredictorJudgeWindowDataset
from safetyjev.visual_dataset import APWindowDataset,collate_numpy


def definitions():
    d=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    d['definitions']=[d['definitions'][0]]
    d['definitions'][0]['label_rule']['threshold_deg']=15
    d['definitions'][0]['scope_bindings']={'jar_transport':['upright']}
    return d


def prepare(tmp_path,task):
    dirs=[];splits={}
    for i,split in enumerate(('train','validation','test')):
        raw=tmp_path/'raw'/str(i);source_fixture(raw,i,semantic=True)
        dirs.append(raw);splits[f'jar_transport/task_{i:04d}']=split
    package=tmp_path/task
    build_package(dirs,package,group_splits=splits,semantic_definitions=definitions(),semantic_task=task,review=True)
    return package


def test_both_loaders_share_original_images_and_never_expose_label_evidence(tmp_path):
    p=prepare(tmp_path,'predictor_judge')
    # Reuse sources; package construction must not copy or modify raw media.
    before=(tmp_path/'raw/0/trajectory.hdf5').stat().st_size
    c=tmp_path/'classifier'
    build_package(sorted((tmp_path/'raw').iterdir()),c,group_splits={f'jar_transport/task_{i:04d}':s for i,s in enumerate(('train','validation','test'))},
        semantic_definitions=definitions(),semantic_task='classifier',review=True)
    jd=PredictorJudgeWindowDataset(p,'train');cd=APWindowDataset(c,'train')
    try:
        assert len(jd)==3 and len(cd)==5
        j=jd[0];a=cd[0]
        assert j['target']==[0.,1.] and a['target'].tolist()==[1.,0.]
        np.testing.assert_array_equal(j['inputs']['observations']['overview'][-1].transpose(1,2,0),a['inputs']['observations']['overview'][0])
        assert set(a['inputs'])=={'question','observations'}
        assert set(j['inputs'])=={'questions','observations','robot_state','remaining_actions','action_mask','action_dt_s','history_mask','constraint_context'}
        assert a['inputs']['observations']['overview'].shape==(1,16,16,3)
        assert collate_numpy([a,cd[1]])['targets'].shape==(2,2)
        assert j['inputs']['action_mask'].tolist()==[True]*8
        assert len(list(p.glob('annotations/*.json')))==3
        assert before==(tmp_path/'raw/0/trajectory.hdf5').stat().st_size
        inv=[json.loads(x) for x in (p/'episode_inventory.jsonl').read_text().splitlines()]
        assert all(x['legacy_safety']=='safe' and x['safety']=='unsafe' and x['selected'] for x in inv)
    finally:jd.close();cd.close()


def test_semantic_package_remains_relocatable_and_annotation_tampering_is_rejected(tmp_path):
    parent=tmp_path/'before';p=prepare(parent,'predictor_judge')
    moved=tmp_path/'after';shutil.move(parent,moved);p=moved/'predictor_judge'
    ds=PredictorJudgeWindowDataset(p,'train')
    try:assert ds[0]['target']==[0.,1.]
    finally:ds.close()
    meta=json.loads((p/'dataset_metadata.json').read_text());res=meta['resources']['source0']
    ap=p/res['annotation_file'];a=json.loads(ap.read_text());a['queries']['excessive_tilt:upright']['question']='Changed question?';ap.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='[Aa]nnotation'):
        ds=PredictorJudgeWindowDataset(p,'train')
        try:ds[0]
        finally:ds.close()


def test_unreviewed_definitions_do_not_create_a_production_package(tmp_path):
    raw=tmp_path/'raw';source_fixture(raw,semantic=True)
    pending=definitions()
    pending['definitions'][0]['definition_status']='review_only'
    with pytest.raises(ValueError,match='review'):
        build_package([raw],tmp_path/'package',semantic_definitions=pending)
    assert not (tmp_path/'package').exists()


def test_heldout_state_changes_cannot_change_training_normalization(tmp_path):
    import h5py
    from safetyjev.source_episodes import load_source_record
    package=prepare(tmp_path,'predictor_judge')
    before=json.loads((package/'dataset_metadata.json').read_text())['normalization']
    train=load_source_record(tmp_path/'raw/0')
    for name,values in [('state',[r['robot_state'] for r in train['observations']]),
                        ('action',[r['command'] for r in train['execution']])]:
        values=np.asarray(values,dtype=np.float64)
        np.testing.assert_allclose(before[name+'_mean'],values.mean(0))
        np.testing.assert_allclose(before[name+'_std'],np.maximum(values.std(0),1e-6),atol=1e-7)
    for task in (1,2):
        with h5py.File(tmp_path/f'raw/{task}/trajectory.hdf5','r+') as f:
            for name in ('q','dq'):
                f['robot/'+name][:]=f['robot/'+name][:]+1000*task
    rebuilt=tmp_path/'rebuilt'
    build_package(sorted((tmp_path/'raw').iterdir()),rebuilt,
        group_splits={f'jar_transport/task_{i:04d}':s for i,s in enumerate(('train','validation','test'))},
        semantic_definitions=definitions(),semantic_task='predictor_judge')
    after=json.loads((rebuilt/'dataset_metadata.json').read_text())
    assert after['normalization']==before
    # Confirm the adversarial records were consumed, not silently excluded.
    for task in (1,2):
        resource=after['resources'][f'source{task}']
        record=json.loads((rebuilt/resource['record_file']).read_text())
        assert record['observations'][0]['robot_state'][0]==1000*task


def test_reuse_annotations_skips_snapshot_extraction_and_preserves_output(tmp_path, monkeypatch):
    from safetyjev import source_episodes
    p=prepare(tmp_path,'predictor_judge')
    kwargs=dict(group_splits={f'jar_transport/task_{i:04d}':s for i,s in enumerate(('train','validation','test'))},
        semantic_definitions=definitions(),semantic_task='classifier',review=True)
    sources=sorted((tmp_path/'raw').iterdir())
    fresh=tmp_path/'fresh';build_package(sources,fresh,**kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reuse must not reread physical snapshots')
    monkeypatch.setattr(source_episodes,'extract_semantic_evidence',forbidden)
    reused=tmp_path/'reused'
    build_package(sources,reused,reuse_annotations=p/'annotations',**kwargs)
    for path in (reused/'annotations').glob('*.json'):
        assert path.stat().st_ino==(p/'annotations'/path.name).stat().st_ino
    for split in ('train','validation','test','excluded'):
        assert (fresh/(split+'.jsonl')).read_bytes()==(reused/(split+'.jsonl')).read_bytes()
    assert [(f.name,f.read_bytes()) for f in sorted((fresh/'annotations').glob('*.json'))]==[
        (f.name,f.read_bytes()) for f in sorted((reused/'annotations').glob('*.json'))]
    # The delivered package owns its small annotations, not a dependency on p.
    shutil.rmtree(p)
    ds=APWindowDataset(reused,'train')
    try:assert len(ds)==5 and ds[0]['target'].tolist()==[1.,0.]
    finally:ds.close()


@pytest.mark.parametrize('change',['source','definition','annotation','missing'])
def test_reuse_annotations_rejects_stale_or_missing_inputs(tmp_path,change):
    import h5py
    p=prepare(tmp_path,'predictor_judge');d=definitions()
    annotations=p/'annotations';file=next(annotations.glob('*.json'))
    if change=='source':
        with h5py.File(tmp_path/'raw/0/trajectory.hdf5','r+') as f:
            f['objects/pose_world'][0,0,2]=100.
    elif change=='definition':d['definitions'][0]['label_rule']['threshold_deg']=30
    elif change=='annotation':file.write_text(file.read_text().replace('excessively tilted','upright'))
    else:file.unlink()
    with pytest.raises((ValueError,FileNotFoundError),match='[Ss]ource|[Dd]efinition|[Dd]igest|No such file'):
        build_package(sorted((tmp_path/'raw').iterdir()),tmp_path/'reused',
            group_splits={f'jar_transport/task_{i:04d}':s for i,s in enumerate(('train','validation','test'))},
            semantic_definitions=d,semantic_task='classifier',review=True,reuse_annotations=annotations)


def test_annotation_reuse_requires_explicit_semantic_mode(tmp_path):
    with pytest.raises(ValueError,match='Semantic'):
        build_package([],tmp_path/'package',reuse_annotations=tmp_path/'annotations')
    assert not (tmp_path/'package').exists()


def test_caches_preserve_semantic_inputs_and_reject_changed_labels(tmp_path):
    from safetyjev.visual_cache import build_visual_cache
    from safetyjev.predictor_judge_cache import build_cache
    from safetyjev.visual_train import verify_package
    for task,loader,cache_builder in [('classifier',APWindowDataset,build_visual_cache),('predictor_judge',PredictorJudgeWindowDataset,build_cache)]:
        package=prepare(tmp_path/task,task);cache=tmp_path/(task+'-cache')
        verify_package(package)
        cache_builder(package,cache)
        direct=loader(package,'train');cached=loader(package,'train',frame_cache=cache)
        try:
            np.testing.assert_array_equal(direct[0]['inputs']['observations']['overview'],cached[0]['inputs']['observations']['overview'])
            np.testing.assert_array_equal(direct[0]['target'],cached[0]['target'])
        finally:direct.close();cached.close()
        # Edited question/label bytes invalidate cached sample offsets too.
        p=package/'train.jsonl';rows=p.read_text().splitlines();r=json.loads(rows[0]);r['target']=[1.,0.] if r['target']==[0.,1.] else [0.,1.];rows[0]=json.dumps(r);p.write_text('\n'.join(rows)+'\n')
        with pytest.raises(ValueError,match='[Ss]plit'):
            loader(package,'train',frame_cache=cache)


@pytest.mark.parametrize('task',['classifier','predictor_judge'])
def test_cached_consumers_reject_mutated_annotations(tmp_path,task):
    from safetyjev.visual_cache import build_visual_cache
    from safetyjev.predictor_judge_cache import build_cache
    package=prepare(tmp_path,task);cache=tmp_path/'cache'
    loader=APWindowDataset if task=='classifier' else PredictorJudgeWindowDataset
    (build_visual_cache if task=='classifier' else build_cache)(package,cache)
    meta=json.loads((package/'dataset_metadata.json').read_text())
    path=package/meta['resources']['source0']['annotation_file']
    path.write_text(path.read_text()+' ')
    ds=loader(package,'train',frame_cache=cache)
    try:
        with pytest.raises(ValueError,match='Annotation'):ds[0]
    finally:ds.close()


def test_missing_query_evidence_does_not_make_heldout_episode_fully_safe(tmp_path):
    import h5py
    raw_train=tmp_path/'raw/0';source_fixture(raw_train,0,semantic=True)
    raw_test=tmp_path/'raw/1';source_fixture(raw_test,1,semantic=True)
    with h5py.File(raw_test/'trajectory.hdf5','r+') as f:
        poses=f['objects/pose_world'][:]
        poses[:,:,3:]=[0,0,0,1]
        poses[2,:,3:]=[0,0,0,0] # One unusable orientation; other steps are upright.
        f['objects/pose_world'][:]=poses
    p=tmp_path/'package'
    build_package([raw_train,raw_test],p,group_splits={'jar_transport/task_0000':'train','jar_transport/task_0001':'test'},
        semantic_definitions=definitions(),semantic_task='classifier',review=True)
    inventory=[json.loads(s) for s in (p/'episode_inventory.jsonl').read_text().splitlines()]
    entry=next(r for r in inventory if r['episode_id']=='source1')
    assert entry['safety']=='unknown' and entry['selected']
    assert entry['semantic_coverage']['unavailable_state_observations']==1
    samples=[json.loads(s) for s in (p/'test.jsonl').read_text().splitlines()]
    assert len(samples)==4 and all(s['target']==[1.,0.] for s in samples)


def test_reviewed_liquid_build_load_and_reuse(tmp_path):
    import h5py
    from maniguard.data.recording.schema import append_blob,pack_record
    from test_liquid_geometry import encrypted_asset
    pytest.importorskip('pxr.Usd')
    assets=tmp_path/'assets';assets.mkdir();scene,_=encrypted_asset(assets)
    init=scene['objects_info']['init_info'];init['jar_1']=init.pop('cup_1');init['jar_1']['args'].update(name='jar_1',scale=[2,2,2])
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,planned_steps=4)
    (raw/'task_scene.json').write_text(json.dumps(scene))
    mp=raw/'episode.json';m=json.loads(mp.read_text());m['snapshots']=5
    m['proposition_bindings']['liquid_spilled']={'over':[{'name':'jar_1','category':'jar'}]};mp.write_text(json.dumps(m))
    with h5py.File(raw/'snapshots.hdf5','w') as f:
        f.create_dataset('observation_id',data=np.arange(5))
        for t in range(5):
            xyz=np.zeros((100,3));xyz[:20 if t>=2 else 0]=[10,10,10]
            state={'pos':np.zeros(3),'ori':np.array([0,0,0,1]),'registry':{'system_registry':{'water':{'particle_states':{'waterInstancer0':{'n_particles':100,'particle_positions':xyz}}}}}}
            append_blob(f,'records',pack_record({'state':state,'checkpoint':None}))
    d=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    d['definitions']=[next(x for x in d['definitions'] if x['semantic_id']=='liquid_spill')]
    d['definitions'][0]['scope_bindings']={'jar_transport':['liquid_spilled']}
    package=tmp_path/'package'
    kwargs=dict(group_splits={'jar_transport/task_0000':'train'},semantic_definitions=d)
    build_package([raw],package,liquid_asset_root=assets,**kwargs)
    ds=PredictorJudgeWindowDataset(package,'train')
    try:
        assert len(ds)==4
        assert [ds[i]['target'] for i in range(4)]==[[0.,1.],[0.,1.],[1.,0.],[1.,0.]]
        assert ds[0]['inputs']['action_mask'].tolist()==[True]*4+[False]*4
        assert '20%' not in ds[0]['inputs']['questions'][0]
    finally:ds.close()
    reused=tmp_path/'reuse';build_package([raw],reused,reuse_annotations=package/'annotations',**kwargs)
    assert (package/'train.jsonl').read_bytes()==(reused/'train.jsonl').read_bytes()
