import hashlib,json
import pytest


def test_manifest_selects_only_explicit_node_paths_and_checks_identity(tmp_path):
    from safetyjev.source_manifest import read_source_manifest
    raw=tmp_path/'raw';raw.mkdir();content={'episode_id':'e'}
    (raw/'episode.json').write_text(json.dumps(content))
    sha=hashlib.sha256((raw/'episode.json').read_bytes()).hexdigest()
    rows=[dict(episode_id='e',group_id='jar/task_0000',source_node='n1',source_path=str(raw),
               source_metadata_sha256=sha,split='train'),
          dict(episode_id='remote',group_id='jar/task_0001',source_node='n2',source_path='/not-local',split='test')]
    path=tmp_path/'manifest.json';path.write_text(json.dumps(rows))
    result=read_source_manifest(path,node='n1',group_splits={'jar/task_0000':'train','jar/task_0001':'test'})
    assert result['directories']==[raw]
    assert result['source_manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['selected'][0]['source_node']=='n1'
    with pytest.raises(ValueError,match='split'):read_source_manifest(path,node='n1',group_splits={'jar/task_0000':'test'})
    content['episode_id']='changed';(raw/'episode.json').write_text(json.dumps(content))
    with pytest.raises(ValueError,match='identity'):read_source_manifest(path,node='n1',group_splits={'jar/task_0000':'train'})


def test_global_normalization_combines_weighted_training_statistics_only():
    import numpy as np
    from safetyjev.source_manifest import combine_training_normalization
    def stats(values):
        x=np.asarray(values,dtype=float)
        return dict(sum=x.sum(0).tolist(),sum_squares=(x*x).sum(0).tolist(),count=len(x))
    shards=[]
    for values in ([[0.,2.]],[[4.,6.],[8.,10.],[12.,14.]]):
        shards.append(dict(normalization_statistics={k:stats(values) for k in ('state','action')}))
    result=combine_training_normalization(shards)
    np.testing.assert_allclose(result['state_mean'],[6,8])
    np.testing.assert_allclose(result['state_std'],np.sqrt(20))
    # Averaging local standard deviations or equally weighting shard means is wrong.
    assert result['action_mean']==result['state_mean']
    with pytest.raises(ValueError):combine_training_normalization([{}])


def test_explicit_partition_preserves_source_split_and_checks_frozen_bytes(tmp_path):
    from safetyjev.source_manifest import read_source_manifest
    raw=tmp_path/'raw';raw.mkdir()
    meta=b'{"episode_id":"e"}';(raw/'episode.json').write_bytes(meta)
    row=dict(run_id='r',episode_id='e',group_id='jar/task_0000',source_node='n1',
             source_path=str(raw),source_metadata_sha256=hashlib.sha256(meta).hexdigest(),split='test')
    path=tmp_path/'source.json';path.write_text(json.dumps([row]))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    options=dict(node='n1',group_splits={'jar/task_0000':'train'},allow_split_override=True,
                 expected_sha256=digest,expected_count=1)
    result=read_source_manifest(path,**options)
    assert result['selected'][0]['split']=='test'
    assert result['build_group_splits']=={'jar/task_0000':'train'}
    assert result['split_overrides']==[dict(episode_id='e',group_id='jar/task_0000',
                                           source_split='test',build_split='train')]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with pytest.raises(ValueError,match='count'):
        read_source_manifest(path,**{**options,'expected_count':2})
    path.write_text(json.dumps([row])+'\n')
    with pytest.raises(ValueError,match='SHA256'):
        read_source_manifest(path,**options)


def test_source_manifest_rejects_duplicate_runs_even_on_other_nodes(tmp_path):
    from safetyjev.source_manifest import read_source_manifest
    rows=[dict(run_id='same',episode_id=str(i),source_node='remote',source_path='/raw/'+str(i),
               group_id='jar/task_0000',split='train') for i in range(2)]
    path=tmp_path/'source.json';path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Duplicate'):
        read_source_manifest(path,node='local',group_splits={'jar/task_0000':'train'})
