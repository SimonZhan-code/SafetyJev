import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest


def test_training_media_preserves_jpeg_and_rejects_changed_payload(tmp_path):
    from safetyjev.training_media import export_media
    from safetyjev.source_episodes import MediaReader
    raw=tmp_path/'raw';raw.mkdir()
    blobs=[b'jpeg-first',b'jpeg-second']
    with h5py.File(raw/'trajectory.hdf5','w') as f:
        g=f.create_group('images/view');g['bytes']=np.frombuffer(b''.join(blobs),dtype='u1');g['offsets']=[0,len(blobs[0]),sum(map(len,blobs))]
        f['supervision']=[1,0]
    refs={f'hdf5:trajectory.hdf5:view:{i}':hashlib.sha256(b).hexdigest() for i,b in enumerate(blobs)}
    target=tmp_path/'training/trajectory.hdf5'
    result=export_media(raw,refs,target)
    assert result['images']==2
    with h5py.File(target) as f:assert list(f)==['images']
    with MediaReader(target.parent) as reader:
        for ref,digest in refs.items():assert reader.read(ref,digest) in blobs
    assert export_media(raw,refs,target)==result
    with h5py.File(target,'r+') as f:f['images/view/bytes'][0]=0
    with pytest.raises(ValueError,match='bytes changed'):export_media(raw,refs,target)


def test_both_models_relocate_and_cache_without_any_raw_files(tmp_path):
    from test_source_episodes import source_fixture
    from test_semantic_package import definitions
    from safetyjev.predictor_judge_dataset import build_package,PredictorJudgeWindowDataset
    from safetyjev.visual_dataset import APWindowDataset
    from safetyjev.package_layout import publish_node_package,combine_package_shards
    from safetyjev.training_media import publish_training_media
    from safetyjev.predictor_judge_cache import build_cache as judge_cache
    from safetyjev.visual_cache import build_visual_cache as visual_cache
    root=tmp_path/'delivery';root.mkdir()
    splits={f'jar_transport/task_{i:04d}':'train' for i in range(2)}
    for i in range(2):
        raw=root/'sources'/str(i)/f'source{i}';source_fixture(raw,i,semantic=True,observation_count=9)
        for task in ('predictor_judge','classifier'):
            package=tmp_path/f'build-{task}-{i}'
            build_package([raw],package,group_splits=splits,semantic_definitions=definitions(),semantic_task=task,input_contract='chunk_start_v2' if task=='predictor_judge' else 'per_step_v1',train_episodes='all')
            publish_node_package(package,root,task,str(i),{f'source{i}':f'sources/{i}/source{i}'})
        report=publish_training_media(root,str(i),workers=2)
        assert report['episodes']==1 and report['images']==18
        assert publish_training_media(root,str(i))==report
    shutil.rmtree(root/'sources') # The downloaded training subset has no raw backup.
    moved=tmp_path/'download';shutil.move(root,moved)
    for task,cls,cache in [('predictor_judge',PredictorJudgeWindowDataset,judge_cache),('classifier',APWindowDataset,visual_cache)]:
        summary=combine_package_shards(moved/task,['0','1'])
        assert set(summary['resources'])=={'source0','source1'}
        ds=cls(moved/task,'train')
        try:
            assert ds[0]['target'] is not None and ds[len(ds)-1]['target'] is not None
        finally:ds.close()
        cache_root=tmp_path/(task+'-cache');cache(moved/task,cache_root)
        ds=cls(moved/task,'train',frame_cache=cache_root)
        try:assert ds[len(ds)-1]['target'] is not None
        finally:ds.close()
