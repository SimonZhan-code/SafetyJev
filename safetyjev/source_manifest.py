"""Explicit immutable source selection; never discover extra campaign episodes."""
import hashlib,json
from pathlib import Path


def read_source_manifest(path,*,node,group_splits,allow_split_override=False,
                         expected_sha256=None,expected_count=None):
    raw=Path(path).read_bytes();rows=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Source manifest SHA256 differs')
    if not isinstance(rows,list) or not rows:raise ValueError('Nonempty source manifest required')
    if expected_count is not None and len(rows)!=expected_count:raise ValueError('Source manifest count differs')
    seen=set();runs=set();locations=set();selected=[];directories=[];overrides=[]
    for row in rows:
        eid=row['episode_id'];location=(str(row['source_node']),row['source_path'])
        if eid in seen or location in locations or (row.get('run_id') is not None and row['run_id'] in runs):
            raise ValueError('Duplicate source identity')
        seen.add(eid);locations.add(location)
        if row.get('run_id') is not None:runs.add(row['run_id'])
        build_split=group_splits.get(row['group_id'])
        if build_split not in ('train','validation','test') or row['split'] not in ('train','validation','test'):
            raise ValueError('Missing or invalid explicit split assignment')
        if build_split!=row['split']:
            if not allow_split_override:
                raise ValueError('Manifest/source split differs from explicit build assignment')
            overrides.append(dict(episode_id=eid,group_id=row['group_id'],source_split=row['split'],build_split=build_split))
        if str(row['source_node'])!=str(node):continue
        root=Path(row['source_path'])
        if not root.is_absolute():raise ValueError('Absolute source path required')
        metadata=(root/'episode.json').read_bytes();m=json.loads(metadata)
        if m.get('episode_id')!=eid or hashlib.sha256(metadata).hexdigest()!=row['source_metadata_sha256']:
            raise ValueError('Source metadata identity changed')
        selected.append(row);directories.append(root)
    if not selected:raise ValueError('No sources for requested node')
    return dict(directories=directories,selected=selected,source_manifest_sha256=digest,
                source_manifest_episodes=len(rows),source_node=str(node),
                build_group_splits=dict(group_splits),split_overrides=overrides)


def combine_training_normalization(metadata):
    """Combine train-only sufficient statistics, never average shard standard deviations."""
    import numpy as np
    if not metadata:raise ValueError('Training shard statistics required')
    result={}
    for key in ('state','action'):
        total=square=None;count=0
        for shard in metadata:
            stats=shard.get('normalization_statistics',{}).get(key)
            if stats is None:raise ValueError('Missing train normalization statistics')
            n=stats['count'];s=np.asarray(stats['sum'],dtype=float);q=np.asarray(stats['sum_squares'],dtype=float)
            if type(n) is not int or n<=0 or s.ndim!=1 or q.shape!=s.shape or not np.isfinite(s).all() or not np.isfinite(q).all() or (q<0).any():
                raise ValueError('Invalid training statistics')
            if total is None:total=np.zeros_like(s);square=np.zeros_like(q)
            if total.shape!=s.shape:raise ValueError('Training feature dimensions differ')
            total+=s;square+=q;count+=n
        mean=total/count;std=np.maximum(np.sqrt(np.maximum(square/count-mean*mean,0)),1e-6)
        result[key+'_mean']=mean.tolist();result[key+'_std']=std.tolist()
    return result
