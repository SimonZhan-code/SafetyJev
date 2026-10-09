"""Explicit immutable source selection; never discover extra campaign episodes."""
import hashlib,json
from pathlib import Path


def read_source_manifest(path,*,node,group_splits):
    raw=Path(path).read_bytes();rows=json.loads(raw)
    if not isinstance(rows,list) or not rows:raise ValueError('Nonempty source manifest required')
    seen=set();locations=set();selected=[];directories=[]
    for row in rows:
        eid=row['episode_id'];location=(str(row['source_node']),row['source_path'])
        if eid in seen or location in locations:raise ValueError('Duplicate source identity')
        seen.add(eid);locations.add(location)
        if str(row['source_node'])!=str(node):continue
        if group_splits.get(row['group_id'])!=row['split']:
            raise ValueError('Manifest/source split differs from explicit build assignment')
        root=Path(row['source_path'])
        if not root.is_absolute():raise ValueError('Absolute source path required')
        metadata=(root/'episode.json').read_bytes();m=json.loads(metadata)
        if m.get('episode_id')!=eid or hashlib.sha256(metadata).hexdigest()!=row['source_metadata_sha256']:
            raise ValueError('Source metadata identity changed')
        selected.append(row);directories.append(root)
    if not selected:raise ValueError('No sources for requested node')
    return dict(directories=directories,selected=selected,source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                source_manifest_episodes=len(rows),source_node=str(node))


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
