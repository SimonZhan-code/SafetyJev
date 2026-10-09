import hashlib,json,shutil
import pytest


def test_sharded_split_has_contiguous_seek_offsets_and_content_hash(tmp_path):
    from safetyjev.package_layout import split_file
    from safetyjev.predictor_judge_dataset import file_hash
    data=[b'{"a":1}\n',b'',b'{"b":2}\n{"c":3}\n']
    for i,b in enumerate(data):(tmp_path/f'{i}.jsonl').write_bytes(b)
    summary={'split_files':{'train':[f'{i}.jsonl' for i in range(3)]}}
    path=split_file(tmp_path,'train',summary)
    assert path.stat().st_size==sum(map(len,data))
    assert file_hash(path)==hashlib.sha256(b''.join(data)).hexdigest()
    with path.open('rb') as f:
        assert f.readline()==data[0]
        offset=f.tell();assert f.readline()==b'{"b":2}\n'
        f.seek(offset);assert f.read(4)==b'{"b"'
        assert f.read()==b':2}\n{"c":3}\n'
        f.seek(0);assert list(f)==b''.join(data).splitlines(keepends=True)
    with pytest.raises(ValueError,match='escapes'):
        split_file(tmp_path,'train',{'split_files':{'train':['../other']}})


def test_joint_shards_and_shared_resources_relocate_and_both_caches_read(tmp_path):
    from test_semantic_package import definitions
    from test_source_episodes import source_fixture
    from safetyjev.predictor_judge_dataset import build_package,PredictorJudgeWindowDataset
    from safetyjev.visual_dataset import APWindowDataset
    from safetyjev.package_layout import publish_node_package,combine_package_shards
    from safetyjev.predictor_judge_cache import build_cache as judge_cache
    from safetyjev.visual_cache import build_visual_cache as visual_cache
    root=tmp_path/'delivery';root.mkdir()
    assignment={f'jar_transport/task_{i:04d}':'train' for i in range(2)}
    for i in range(2):
        raw=root/'sources'/str(i)/f'source{i}';source_fixture(raw,i,semantic=True,observation_count=17)
        for task in ('predictor_judge','classifier'):
            package=tmp_path/f'build-{i}-{task}'
            build_package([raw],package,group_splits=assignment,semantic_definitions=definitions(),semantic_task=task,
                          input_contract='chunk_start_v2' if task=='predictor_judge' else 'per_step_v1',train_episodes='all')
            publish_node_package(package,root,task,str(i),{f'source{i}':f'sources/{i}/source{i}'})
    for task in ('predictor_judge','classifier'):
        combine_package_shards(root/task,['0','1'])
        report=json.loads((root/task/'composition.json').read_text())
        assert report['episodes']['all']['total']==2
        assert sum(r['eligible'] for r in report['query_windows'])==(4 if task=='predictor_judge' else 34)
        assert len((root/task/'episode_inventory.jsonl').read_text().splitlines())==2
    assert len(list((root/'shared/annotations').rglob('*.json')))==2
    moved=tmp_path/'moved';shutil.move(root,moved)
    for task,cls,cache in [('predictor_judge',PredictorJudgeWindowDataset,judge_cache),('classifier',APWindowDataset,visual_cache)]:
        package=moved/task;ds=cls(package,'train')
        try:
            first,last=ds[0],ds[len(ds)-1]
            assert first['target'] is not None and last['target'] is not None
            assert len(ds)==(4 if task=='predictor_judge' else 34)
        finally:ds.close()
        cache_root=tmp_path/(task+'-cache');cache(package,cache_root)
        ds=cls(package,'train',frame_cache=cache_root)
        try:assert ds[len(ds)-1]['target'] is not None
        finally:ds.close()


def test_changed_shard_and_cross_node_protocol_are_rejected(tmp_path):
    from test_semantic_package import prepare
    from safetyjev.package_layout import publish_node_package,combine_package_shards
    root=tmp_path/'delivery';root.mkdir()
    package=prepare(tmp_path/'build','predictor_judge')
    sources={f'source{i}':f'sources/0/source{i}' for i in range(3)}
    publish_node_package(package,root,'predictor_judge','0',sources)
    shard=root/'predictor_judge/shards/0/train.jsonl'
    with shard.open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='bytes changed'):
        combine_package_shards(root/'predictor_judge',['0'])


def test_changed_inventory_is_not_accepted_as_source_provenance(tmp_path):
    from test_semantic_package import prepare
    from safetyjev.package_layout import publish_node_package,combine_package_shards
    root=tmp_path/'delivery';root.mkdir();package=prepare(tmp_path/'build','predictor_judge')
    publish_node_package(package,root,'predictor_judge','0',{f'source{i}':f'sources/0/source{i}' for i in range(3)})
    with (root/'predictor_judge/shards/0/episode_inventory.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='inventory'):
        combine_package_shards(root/'predictor_judge',['0'])


def test_positive_segments_are_not_window_counts_or_uncensored_physical_onsets():
    from safetyjev.semantic_reporting import state_segments
    a={'episode_id':'e','queries':{'q':{'temporal_kind':'state','semantic_id':'tilt',
       'states':[dict(step=i,label=v) for i,v in enumerate([1,1,0,1,None,1,0,1])]},
       'liquid':{'temporal_kind':'interval','states':[]}}}
    rows=state_segments(a)
    assert len(rows)==1
    assert rows[0]['positive_segments']==4
    assert rows[0]['observed_safe_to_unsafe_onsets']==2
    assert rows[0]['left_censored_segments']==1
    assert rows[0]['unknown_prior_segments']==1


def test_publication_rejects_changed_original_inventory(tmp_path):
    from test_semantic_package import prepare
    from safetyjev.package_layout import publish_node_package
    package=prepare(tmp_path/'build','predictor_judge')
    with (package/'episode_inventory.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='inventory'):
        publish_node_package(package,tmp_path/'delivery','predictor_judge','0',
                             {f'source{i}':f'sources/0/source{i}' for i in range(3)})


def test_state_segments_do_not_join_or_infer_onsets_across_reset():
    from safetyjev.semantic_reporting import state_segments
    a={'episode_id':'e','reset_steps':[1,3],'queries':{'q':{'temporal_kind':'state','semantic_id':'tilt',
       'states':[dict(step=i,label=v) for i,v in enumerate([1,1,0,1])]}}}
    row=state_segments(a)[0]
    assert row['positive_segments']==3
    assert row['unknown_prior_segments']==2
    assert row['observed_safe_to_unsafe_onsets']==0
