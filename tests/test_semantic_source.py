"""Offline physical evidence, using the actual source HDF5 schema."""
import hashlib
import json
import sys

import numpy as np
import pytest

from safetyjev import source_episodes


def extract(path, **kwargs):
    assert hasattr(source_episodes, 'extract_semantic_evidence'), 'Semantic source extraction is not implemented'
    return source_episodes.extract_semantic_evidence(path, **kwargs)


@pytest.fixture
def semantic_source(tmp_path):
    from maniguard.data.recording.writer import EpisodeWriter

    path = tmp_path / 'raw'
    names = ['agent_0', 'jar_1', 'table_0']
    aps = ['jar_upright', 'jar_dropped', 'liquid_spilled', 'food_touched_by_agent',
           'jar_closed', 'jar_on_support']
    metadata = dict(episode_id='synthetic_jar', camera_keys=['image_left', 'wrist_image'],
        resolution=[16, 16], rgb_encoding=dict(codec='jpeg', quality=95, subsampling='444', color_order='RGB'),
        action_hz=20, physics_hz=120, object_names=names, ap_names=aps,
        scene=dict(pipeline='jar_transport', surface_name='table_0',prompt='Carry the jar.',target_name='jar_1'),
        proposition_bindings={a: dict(over=[dict(name='jar_1', category='jar')]) for a in aps})
    with EpisodeWriter(path, metadata, max_output_bytes=10_000_000) as writer:
        for t, angle in enumerate([0, 16, 0]):
            if t:
                writer.append_transition(dict(source_observation_id=t-1, target_observation_id=t,
                    duration_s=.05, raw=np.zeros(8), applied=np.zeros(8)))
            pose = np.array([[0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 1-.1*t, np.sin(np.deg2rad(angle)/2), 0, 0, np.cos(np.deg2rad(angle)/2)],
                             [0, 0, .8, 0, 0, 0, 1]])
            measured = lambda value: dict(valid=True, value=value)
            writer.append_observation(dict(observation_id=t, physics_tick=6*t, sim_time_s=.05*t,
                camera_frames={c:np.zeros((16,16,3), np.uint8) for c in metadata['camera_keys']},
                robot=dict(q=np.zeros(9), dq=np.zeros(9)), objects=dict(pose_world=pose),
                object_details={'jar_1':dict(linear_velocity=measured(np.array([0., 0., -t])),
                    angular_velocity=dict(valid=False, error='sensor unavailable') if t==1 else measured(np.zeros(3)),
                    contacts=measured(['/World/scene_0/agent_0/finger'] if t==1 else []),
                    joints={'lid':measured((np.array([.3]), np.array([0.]), np.array([0.])) )})},
                safety=dict(monitor_valid=True, monitor_rejected=t>=1,
                    ap_values=np.array([t!=1, False, False, t==1, False, t==0]),
                    ap_valid=np.array([True, True, True, t!=2, True, True]))))
            # Total system particles remain 100 even if water leaves a container.
            state = dict(pos=np.zeros(3), ori=np.array([0,0,0,1]), registry=dict(
                object_registry={'agent_0':dict(ag_obj_constraint_params={'0':
                    {'ag_obj_prim_path':'/World/scene_0/jar_1'} if t==0 else {}})},
                system_registry={'water':dict(particle_states={'waterInstancer0':dict(
                    n_particles=100, particle_positions=np.zeros((100,3)))})}),
                _recording_ag_memory={'agent_0':{'_ag_release_counter':{'0':None}}})
            cp = dict(monitor=dict(predicate_memory={'liquid_spilled':dict(initial_counts={'jar_1':100})}))
            if t != 2:
                writer.append_snapshot(t, state, cp)
        writer.finish('complete', {})
    (path/'diagnostics.jsonl').write_text(json.dumps(dict(surface_info=dict(
        frame='world_aabb', top_z=.8, bounds_xy=[[-1,-1],[1,1]])))+'\n')
    return path


def file_hashes(path):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


def test_extraction_preserves_clocks_bindings_values_and_original_files(semantic_source):
    before = file_hashes(semantic_source)
    modules_before = set(sys.modules)
    result = extract(semantic_source)
    assert not any(n.startswith(('omnigibson', 'spot', 'isaacsim')) for n in set(sys.modules)-modules_before)
    assert file_hashes(semantic_source) == before
    assert result['episode_id'] == 'synthetic_jar'
    assert result['family'] == 'jar_transport'
    assert result['task_instruction'] == 'Carry the jar.'
    assert result['target_name'] == 'jar_1'
    assert result['source_observations'] == 3 and result['source_transitions'] == 2
    assert [s['step'] for s in result['observations']] == [0, 1, 2]
    assert result['observations'][1]['sim_time_s'] == .05
    assert result['bindings']['jar_upright']['over'][0]['name'] == 'jar_1'
    obj = result['observations'][1]['objects']['jar_1']
    assert obj['pose_world']['available']
    np.testing.assert_allclose(obj['pose_world']['value'][3], np.sin(np.deg2rad(16)/2))
    assert obj['linear_velocity']['value'] == [0, 0, -1]
    assert obj['contacts']['value'] == ['/World/scene_0/agent_0/finger']
    assert obj['joints']['lid']['available']
    json.dumps(result, allow_nan=False)
    assert result['source_hashes']['episode.json'] == before['episode.json']
    assert result['support_geometry']['diagnostics_reference'] == 'diagnostics.jsonl'
    assert result['evidence_sha256'] == extract(semantic_source)['evidence_sha256']


def test_missing_fields_and_ap_validity_are_not_false_labels(semantic_source):
    rows = extract(semantic_source)['observations']
    assert rows[1]['objects']['jar_1']['angular_velocity'] == dict(
        available=False, value=None, reason='sensor unavailable')
    assert not rows[0]['objects']['table_0']['contacts']['available']
    assert rows[2]['source_ap']['food_touched_by_agent']['value'] is None
    assert not rows[2]['source_ap']['food_touched_by_agent']['available']
    assert rows[0]['source_ap']['jar_dropped']['value'] is False


def test_snapshot_grasp_and_liquid_provenance_cannot_be_confused_with_containment(semantic_source):
    rows = extract(semantic_source)['observations']
    assert rows[0]['assisted_grasp']['available']
    assert rows[0]['assisted_grasp']['value']['agent_0']['0']['ag_obj_prim_path'].endswith('/jar_1')
    assert rows[1]['assisted_grasp']['value']['agent_0']['0'] == {}
    assert not rows[2]['assisted_grasp']['available']
    liquid = rows[0]['liquid']
    assert liquid['initial_counts'] == {'jar_1':100}
    assert liquid['system_particle_counts'] == {'water':100}
    assert liquid['contained_counts']['value'] is None
    assert liquid['contained_counts']['reason'] == 'offline_containment_geometry_not_resolved'
    assert liquid['snapshot_reference']['observation_id'] == 0
    assert liquid['particle_coordinate_frame'] == 'scene'
    assert rows[2]['liquid']['contained_counts']['reason'] == 'snapshot_not_recorded'


def test_bounded_sampling_does_not_hide_missing_final_observation(semantic_source):
    result = extract(semantic_source, observation_ids=[0, 2])
    assert [r['step'] for r in result['observations']] == [0, 2]
    for ids in ([0, 0], [3], [-1], [1, 0]):
        with pytest.raises(ValueError, match='indices'):
            extract(semantic_source, observation_ids=ids)
    p = semantic_source/'episode.json'
    meta = json.loads(p.read_text()); meta['observations'] = 2; p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='final observation'):
        extract(semantic_source, observation_ids=[0])


def test_no_silent_object_name_alias_or_nonfinite_pose(semantic_source):
    import h5py
    with h5py.File(semantic_source/'trajectory.hdf5', 'r+') as f:
        f['objects/pose_world'][0, 1, 0] = np.nan
    p = semantic_source/'episode.json'
    meta=json.loads(p.read_text());meta['proposition_bindings']['jar_upright']['over'][0]['name']='jar_1_0'
    p.write_text(json.dumps(meta))
    result=extract(semantic_source)
    assert result['unresolved_bindings'] == ['jar_1_0']
    assert result['observations'][0]['objects']['jar_1']['pose_world']['reason'] == 'nonfinite_measurement'
    assert 'jar_1_0' not in result['observations'][0]['objects']


def test_optional_monitor_checkpoint_does_not_erase_physical_snapshot(semantic_source):
    import h5py
    from maniguard.data.recording.schema import read_blob, unpack_record, pack_record, append_blob
    with h5py.File(semantic_source/'snapshots.hdf5', 'r+') as f:
        records=[unpack_record(read_blob(f['records'], i)) for i in range(len(f['observation_id']))]
        records[0]['checkpoint'] = None
        del f['records']
        for record in records:
            append_blob(f, 'records', pack_record(record))
    row = extract(semantic_source, observation_ids=[0])['observations'][0]
    assert row['assisted_grasp']['available']
    assert row['liquid']['initial_counts'] is None
    assert row['liquid']['system_particle_counts']['water'] == 100
    assert not row['liquid']['contained_counts']['available']


def test_benchmark_family_takes_precedence_over_shared_generation_pipeline(semantic_source):
    p=semantic_source/'episode.json';m=json.loads(p.read_text())
    m['scene'].update(pipeline='liquid_transport',scene_file='/bench/clutter_pickup/task_0048/target/scene_ep1.json')
    p.write_text(json.dumps(m))
    assert extract(semantic_source,observation_ids=[0])['family']=='clutter_pickup'


def test_recorded_world_support_plane_is_extracted_and_bad_frame_not_guessed(semantic_source):
    e=extract(semantic_source,observation_ids=[0])
    assert e['support_geometry']['top_z']['value']==.8
    p=semantic_source/'diagnostics.jsonl';d=json.loads(p.read_text())
    d['surface_info']['frame']='object_local';p.write_text(json.dumps(d)+'\n')
    assert not extract(semantic_source,observation_ids=[0])['support_geometry']['top_z']['available']


def test_usable_rectangle_retains_recorded_world_height_and_conflicts_are_excluded(semantic_source):
    p=semantic_source/'diagnostics.jsonl'
    d=json.loads(p.read_text());d['surface_info']['frame']='world_usable_rect'
    p.write_text(json.dumps(d)+'\n')
    assert extract(semantic_source,observation_ids=[0])['support_geometry']['top_z']['value']==.8
    # The usable rectangle changes XY placement bounds, not the height frame.
    other=json.loads(json.dumps(d));other['surface_info']['top_z']=.9
    p.write_text(json.dumps(d)+'\n'+json.dumps(other)+'\n')
    result=extract(semantic_source,observation_ids=[0])['support_geometry']['top_z']
    assert result['value'] is None and result['reason']=='conflicting_surface_heights'
    for height in (None, float('nan'), True):
        d['surface_info']['top_z']=height;p.write_text(json.dumps(d)+'\n')
        assert not extract(semantic_source,observation_ids=[0])['support_geometry']['top_z']['available']


def test_validated_container_geometry_recovers_actual_counts(semantic_source):
    from safetyjev.liquid_geometry import ConvexContainerVolume
    import itertools
    import h5py
    from maniguard.data.recording.schema import read_blob,unpack_record,pack_record,append_blob
    with h5py.File(semantic_source/'snapshots.hdf5','r+') as f:
        records=[unpack_record(read_blob(f['records'],i)) for i in range(len(f['observation_id']))]
        particles=records[1]['state']['registry']['system_registry']['water']['particle_states']['waterInstancer0']['particle_positions']
        particles[:2]=[0,0,4]
        del f['records']
        for r in records:append_blob(f,'records',pack_record(r))
    before=file_hashes(semantic_source)
    volume=ConvexContainerVolume([dict(points=list(itertools.product((-2.,2.),repeat=3)),mesh_to_body=np.eye(4))])
    rows=extract(semantic_source,liquid_volumes={'jar_1':volume})['observations']
    assert rows[0]['liquid']['contained_counts']['value']=={'jar_1':100}
    assert rows[1]['liquid']['contained_counts']['value']=={'jar_1':98}
    assert rows[1]['liquid']['system_particle_counts']=={'water':100}
    assert rows[1]['source_ap']['liquid_spilled']['value'] is False
    assert rows[1]['liquid']['geometry_sha256']=={'jar_1':volume.sha256}
    assert rows[2]['liquid']['contained_counts']['reason']=='snapshot_not_recorded'
    assert file_hashes(semantic_source)==before


def test_missing_container_geometry_does_not_guess_from_total_particles(semantic_source):
    row=extract(semantic_source,observation_ids=[0],liquid_volumes={})['observations'][0]
    assert row['liquid']['contained_counts']['value'] is None
    assert row['liquid']['contained_counts']['reason']=='container_geometry_not_available'


def test_preparation_resolves_recorded_assets_and_reuses_counts_without_assets(semantic_source,tmp_path):
    from safetyjev.semantic_data import prepare_semantic_annotation,write_annotation,evaluate_semantic_interval
    from pathlib import Path
    from test_liquid_geometry import encrypted_asset
    pytest.importorskip('pxr.Usd')
    assets=tmp_path/'assets';assets.mkdir()
    scene,_=encrypted_asset(assets)
    init=scene['objects_info']['init_info'];init['jar_1']=init.pop('cup_1');init['jar_1']['args'].update(name='jar_1',scale=[2,2,2])
    (semantic_source/'task_scene.json').write_text(json.dumps(scene))
    definitions=json.loads((Path(__file__).parents[1]/'configs/data/semantic_safety.json').read_text())
    definitions['definitions']=[next(d for d in definitions['definitions'] if d['semantic_id']=='liquid_spill')]
    definitions['definitions'][0]['scope_bindings']={'jar_transport':['liquid_spilled']}
    a=prepare_semantic_annotation(semantic_source,'synthetic_jar',definitions,liquid_asset_root=assets)
    assert a['liquid_assets']['jar_1']['model']=='test'
    assert a['queries']['liquid_spill:liquid_spilled']['states'][0]['contained_counts']['value']=={'jar_1':100}
    assert evaluate_semantic_interval(a,'liquid_spill:liquid_spilled',0,1,task='predictor_judge')['label']==0
    reuse=tmp_path/'annotations';write_annotation(reuse/(hashlib.sha256(b'synthetic_jar').hexdigest()+'.json'),a)
    # Already derived portable supervision does not need access to licensed assets.
    b=prepare_semantic_annotation(semantic_source,'synthetic_jar',definitions,reuse_annotations=reuse)
    assert b['liquid_assets']==a['liquid_assets']
