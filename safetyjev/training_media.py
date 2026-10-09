"""Publish lossless model-input images without requiring the full raw backup."""
import json
import os
from pathlib import Path

from .package_layout import resource_path, verify_package_provenance
from .predictor_judge_dataset import file_hash
from .source_episodes import MediaReader


def export_media(source, references, target):
    """Copy only referenced cameras' original JPEG arrays; retain frame numbering."""
    import h5py
    source=Path(source).resolve();target=Path(target).resolve()
    cameras={}
    for ref in references:
        kind,name,camera,index=ref.split(':')
        if kind!='hdf5' or name!='trajectory.hdf5' or '/' in camera or not camera or int(index)<0:
            raise ValueError('Unsupported training image reference')
        cameras.setdefault(camera,set()).add(int(index))
    if not cameras:raise ValueError('No training image references')
    verified=False
    if target==source/'trajectory.hdf5':
        # A resumed publication already points at its validated training media.
        if not target.exists():raise ValueError('Missing training media')
    elif not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.tmp')
        with h5py.File(source/'trajectory.hdf5','r') as original, h5py.File(temporary,'w') as output:
            for camera,indices in sorted(cameras.items()):
                group=original['images/'+camera]
                if indices!=set(range(len(group['offsets'])-1)):
                    raise ValueError('Training image references must cover the recorded camera timeline')
                dest=output.create_group('images/'+camera)
                for key in ('bytes','offsets'):
                    original.copy(group[key],dest,name=key)
        # Verify every selected image against the already constructed package.
        with MediaReader(temporary.parent) as reader:
            for ref,digest in references.items():
                reader.read(ref.replace(':trajectory.hdf5:',':'+temporary.name+':',1),digest)
        temporary.replace(target)
        verified=True
    with h5py.File(target,'r') as output:
        if set(output)!= {'images'} or set(output['images'])!=set(cameras):
            raise ValueError('Unexpected training media fields')
        for camera,indices in cameras.items():
            group=output['images/'+camera]
            if set(group)!= {'bytes','offsets'} or indices!=set(range(len(group['offsets'])-1)):
                raise ValueError('Training image timeline differs')
    if not verified:
        with MediaReader(target.parent) as reader:
            for ref,digest in references.items():reader.read(ref,digest)
    return dict(images=len(references),bytes=target.stat().st_size,sha256=file_hash(target))


def _export_job(job):
    eid,relative,source,references,target=job
    return eid,relative,export_media(source,references,target)


def publish_training_media(root, node, workers=1):
    """Repoint both unpublished node shards to shared, standalone training media.

    Labels, record bytes, sample indices, splits and normalization are unchanged.
    Source provenance still identifies the original full recording; its HDF5 hash
    is distinct from the new training payload hash.
    """
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=8:
        raise ValueError('Training media workers must be between 1 and 8')
    root=Path(root).resolve()
    if not node or Path(node).name!=node:raise ValueError('Invalid node')
    shards=[]
    for task in ('predictor_judge','classifier'):
        package=root/task/'shards'/node
        if (package/'BUILDING').exists():raise ValueError('Unfinished source package')
        if (root/task/'dataset_metadata.json').exists():
            raise ValueError('Publish training media before combining node shards')
        meta=json.loads((package/'dataset_metadata.json').read_text())
        verify_package_provenance(package,meta)
        shards.append((package,meta))
    if set(shards[0][1]['resources'])!=set(shards[1][1]['resources']):
        raise ValueError('Model episode sets differ')
    def jobs():
        for eid in sorted(shards[0][1]['resources']):
            if Path(eid).name!=eid:raise ValueError('Invalid episode ID')
            package,meta=shards[0];res=meta['resources'][eid]
            other=shards[1][1]['resources'][eid]
            if any(res.get(k)!=other.get(k) for k in ('record_sha256','media_manifest_sha256','annotation_file_sha256')):
                raise ValueError('Model resources differ')
            manifest=resource_path(package,res,'media_manifest')
            if file_hash(manifest)!=res['media_manifest_sha256']:raise ValueError('Image manifest changed')
            record_path=resource_path(package,res,'record_file')
            if file_hash(record_path)!=res['record_sha256']:raise ValueError('Episode record changed')
            record=json.loads(record_path.read_text());references=json.loads(manifest.read_text())
            expected={ref for obs in record['observations'] for ref in obs['images'].values()}
            if set(references)!=expected:raise ValueError('Image reference coverage differs')
            relative=Path('media')/node/eid/'trajectory.hdf5'
            yield eid,relative,resource_path(package,res,'raw_root'),references,root/relative
    def results():
        if workers==1:
            yield from map(_export_job,jobs())
            return
        from concurrent.futures import ProcessPoolExecutor
        from collections import deque
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=deque()
            for job in jobs():
                pending.append(pool.submit(_export_job,job))
                if len(pending)>=workers:yield pending.popleft().result()
            while pending:yield pending.popleft().result()
    files={}
    for eid,relative,receipt in results():
        files[eid]=dict(path=relative.as_posix(),**receipt)
        for p,m in shards:
            r=m['resources'][eid]
            r.update(raw_root=os.path.relpath((root/relative).parent,p),
                     training_media_sha256=receipt['sha256'])
    # Atomic metadata replacement. An interrupted run can resume from either root.
    for package,meta in shards:
        meta['media_delivery']='original_jpeg_training_views'
        temporary=package/'dataset_metadata.training.tmp'
        temporary.write_text(json.dumps(meta,indent=2)+'\n')
        temporary.replace(package/'dataset_metadata.json')
    report=dict(node=node,episodes=len(files),images=sum(f['images'] for f in files.values()),
                bytes=sum(f['bytes'] for f in files.values()),files=files)
    path=root/'reports'/('training-media-'+node+'.json');path.parent.mkdir(exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    return report


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--node',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(argv);report=publish_training_media(a.root,a.node,a.workers)
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':main()
