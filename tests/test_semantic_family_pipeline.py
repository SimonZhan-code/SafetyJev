"""Six-family source-schema acceptance; synthetic geometry is not rollout evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_source_episodes import source_fixture
from test_liquid_geometry import encrypted_asset
from safetyjev.predictor_judge_dataset import build_package, PredictorJudgeWindowDataset
from safetyjev.visual_dataset import APWindowDataset

# Exact AP scope names used by the benchmark, with distinct synthetic objects.
SCOPES = {
    'jar_transport': {'jar_upright':[0], 'jar_dropped':[0], 'jar_on_support':[0], 'jar_closed':[0]},
    'lid_transport': {'container_dropped':[0], 'container_on_support':[0], 'lid_on_container':[1]},
    'stack_retrieve': {'all_stack_upright':[1,2], 'target_upright':[0], 'any_stack_dropped':[1,2], 'target_dropped':[0]},
    'cabinet_pickup': {'all_active_upright':[0,1], 'target_dropped':[0], 'obstacle_dropped':[1]},
    'dusty_transfer': {'food_dropped':[0], 'food_touched_by_agent':[0]},
    'clutter_pickup': {'container_upright':[0], 'obstacles_upright':[1,2], 'container_dropped':[0], 'obstacles_dropped':[1,2], 'liquid_spilled':[0]},
}
CATEGORIES = {
    'jar_transport':['hinged_jar','bowl','bowl'],
    'lid_transport':['tupperware','lid','bowl'],
    'stack_retrieve':['half_french_toast','can_of_bay_leaves','can_of_bay_leaves'],
    'cabinet_pickup':['apricot','jar_of_dill_seed','bowl'],
    'dusty_transfer':['potato','bowl','bowl'],
    'clutter_pickup':['mug','bowl','wineglass'],
}


def family_source(path, family, task, scene):
    import h5py
    from maniguard.data.recording.schema import append_blob,pack_record
    source_fixture(path, task, semantic=True, planned_steps=4)
    mp=path/'episode.json';m=json.loads(mp.read_text())
    m['episode_id']=f'{family}-{task}'
    m['scene']['scene_file']=f'/bench/{family}/task_{task:04d}/base/scene_ep1.json'
    names=['jar_1','other_1','other_2'];m['object_names']=names
    objects=[dict(name=n,category=c) for n,c in zip(names,CATEGORIES[family])]
    bindings={ap:dict(over=[objects[i] for i in scope]) for ap,scope in SCOPES[family].items()}
    m['proposition_bindings'].update(bindings)
    m['ap_names']=['upright',*SCOPES[family]]
    aps=np.ones((5,len(m['ap_names'])),dtype=bool)
    for i,ap in enumerate(m['ap_names']):
        if ap in ('jar_closed','lid_on_container'): aps[:,i]=False
        if ap in ('jar_on_support','container_on_support'): aps[1,i]=False
        if ap=='food_touched_by_agent': aps[:,i]=[False,False,True,False,False]
        if ap=='liquid_spilled': aps[:,i]=[False,False,True,True,True]
    poses=np.zeros((5,3,7),dtype=np.float32);poses[:,:,2]=1;poses[:,:,-1]=1
    q=[np.sin(np.deg2rad(25)/2),0,0,np.cos(np.deg2rad(25)/2)]
    poses[1,0,3:]=q;poses[3,1,3:]=q;poses[2,0,2]=.70
    with h5py.File(path/'trajectory.hdf5','r+') as f:
        for key,data in [('objects/pose_world',poses),('safety/ap_values',aps),('safety/ap_valid',np.ones_like(aps))]:
            del f[key];f.create_dataset(key,data=data)
    (path/'diagnostics.jsonl').write_text(json.dumps({'surface_info':{'frame':'world_aabb','top_z':.8}})+'\n')
    if family=='clutter_pickup':
        (path/'task_scene.json').write_text(json.dumps(scene));m['snapshots']=5
        with h5py.File(path/'snapshots.hdf5','w') as f:
            f.create_dataset('observation_id',data=np.arange(5))
            for t in range(5):
                xyz=np.zeros((100,3));xyz[:20 if t>=2 else 0]=[10,10,10]
                state={'pos':np.zeros(3),'ori':np.array([0,0,0,1]),'registry':{'system_registry':{'water':{'particle_states':{'waterInstancer0':{'n_particles':100,'particle_positions':xyz}}}}}}
                append_blob(f,'records',pack_record({'state':state,'checkpoint':None}))
    mp.write_text(json.dumps(m))


def make_family_packages(root):
    assets=root/'assets';assets.mkdir(parents=True)
    scene,_=encrypted_asset(assets);init=scene['objects_info']['init_info']
    init['jar_1']=init.pop('cup_1');init['jar_1']['args'].update(name='jar_1',scale=[2,2,2])
    definitions=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    for d in definitions['definitions']:
        if d['label_rule']['kind']=='tilt_angle':d['label_rule']['threshold_deg']=15
        if d['label_rule']['kind']=='below_surface':d['label_rule']['margin_m']=.02
    sources=[];splits={}
    for family in SCOPES:
        for task,split in enumerate(('train','validation','test')):
            path=root/'raw'/family/str(task);family_source(path,family,task,scene)
            sources.append(path);splits[f'{family}/task_{task:04d}']=split
    judge=root/'predictor_judge';classifier=root/'classifier'
    build_package(sources,judge,group_splits=splits,semantic_definitions=definitions,liquid_asset_root=assets)
    build_package(sources,classifier,group_splits=splits,semantic_definitions=definitions,
                  semantic_task='classifier',reuse_annotations=judge/'annotations')
    return judge,classifier


def test_six_families_share_packages_without_scope_time_or_split_leaks(tmp_path):
    pytest.importorskip('pxr.Usd')
    judge,classifier=make_family_packages(tmp_path)
    for p,expected in [(judge,76),(classifier,90)]:
        seen=set()
        for split in ('train','validation','test'):
            rows=[json.loads(s) for s in (p/(split+'.jsonl')).read_text().splitlines()]
            assert len(rows)==expected
            meta=[r if p==judge else r['metadata'] for r in rows]
            assert {m['family'] for m in meta}==set(SCOPES)
            groups={r['group_id'] for r in rows};assert not groups&seen;seen|=groups
            assert all(m['definition_sha256'] for m in meta)
        inv=[json.loads(s) for s in (p/'episode_inventory.jsonl').read_text().splitlines()]
        assert len(inv)==18 and all(r['selected'] and r['safety']=='unsafe' for r in inv)
    excluded=[json.loads(s) for s in (classifier/'excluded.jsonl').read_text().splitlines()]
    assert len(excluded)==15
    assert {s['metadata']['label_reason'] for s in excluded}=={'input_history_required'}
    assert {s['metadata']['semantic_id'] for s in excluded}=={'liquid_spill'}
    current=APWindowDataset(classifier,'train');future=PredictorJudgeWindowDataset(judge,'train')
    try:
        assert len(current)==90 and len(future)==76
        for i in range(len(current)):
            item=current[i]
            assert set(item['inputs'])=={'question','observations'}
            assert item['inputs']['observations']['overview'].shape[0]==1
            assert item['target'].tolist() in ([1.,0.],[0.,1.])
        for i in range(len(future)):
            item=future[i]
            assert item['target'] in ([1.,0.],[0.,1.])
            assert set(item['inputs'])=={'questions','observations','robot_state','remaining_actions','action_mask','action_dt_s','history_mask','constraint_context'}
            mask=item['inputs']['action_mask']
            assert 1<=int(mask.sum())<=4
            assert np.count_nonzero(item['inputs']['remaining_actions'][~mask])==0
    finally:current.close();future.close()
    rows=[json.loads(s) for s in (classifier/'train.jsonl').read_text().splitlines()]
    clutter=[r for r in rows if r['metadata']['family']=='clutter_pickup' and r['metadata']['start_step']==1]
    labels={r['metadata']['query_id']:r for r in clutter}
    assert labels['excessive_tilt:container_upright']['target']==[0.,1.]
    assert labels['excessive_tilt:obstacles_upright']['target']==[1.,0.]
    assert labels['excessive_tilt:container_upright']['question']=='Is the mug excessively tilted?'
    assert labels['excessive_tilt:obstacles_upright']['question']=='Is any bowl or wineglass excessively tilted?'


def test_semantic_composition_reconciles_eligible_and_excluded_rows(tmp_path):
    from collections import Counter, defaultdict
    pytest.importorskip('pxr.Usd')
    judge, classifier = make_family_packages(tmp_path)
    for package in (judge, classifier):
        report = json.loads((package/'composition.json').read_text())
        assert 'semantic_windows' in report, 'Missing semantic composition with exclusions'
        summary = (package/'DATA_SUMMARY.md').read_text()
        assert 'first-rejection' not in summary
        assert 'Semantic windows' in summary
        expected = defaultdict(lambda: dict(episodes=set(), positive=0, negative=0, excluded=0,
                                           excluded_reasons=Counter()))
        for filename in ('train', 'validation', 'test', 'excluded'):
            for line in (package/(filename+'.jsonl')).read_text().splitlines():
                row = json.loads(line)
                metadata = row if package == judge else row['metadata']
                entry = expected[(row['split'], metadata['family'], metadata['semantic_id'])]
                entry['episodes'].add(metadata['episode_id'])
                label = 'excluded' if row['target'] is None else ('positive' if row['target'][1] else 'negative')
                entry[label] += 1
                if label == 'excluded':
                    entry['excluded_reasons'][metadata['label_reason']] += 1
        actual = {(r['split'], r['family'], r['semantic_id']):r for r in report['semantic_windows']}
        assert actual.keys() == expected.keys()
        for key, counts in expected.items():
            entry = actual[key]
            assert entry['episodes'] == len(counts['episodes'])
            for label in ('positive', 'negative', 'excluded'):
                assert entry[label] == counts[label]
            assert entry['excluded_reasons'] == dict(counts['excluded_reasons'])
            assert entry['eligible'] == counts['positive'] + counts['negative']
            assert entry['candidates'] == entry['eligible'] + counts['excluded']
        if package == classifier:
            liquid = [r for r in actual.values() if r['semantic_id'] == 'liquid_spill']
            assert len(liquid) == 3
            assert all(r['eligible'] == 0 and r['excluded'] == 5 for r in liquid)
