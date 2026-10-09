"""Bounded parallel validation of immutable source images, shared across tasks."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
from PIL import Image
from .source_episodes import MediaReader


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class MediaReuseError(ValueError):
    pass


def validate_batch(job):
    directory,references=job
    result=[]
    with MediaReader(directory) as reader:
        for relative in references:
            blob=reader.read(relative)
            with Image.open(io.BytesIO(blob)) as im:im.load()
            result.append((relative,hashlib.sha256(blob).hexdigest()))
    return result


class MediaValidator:
    def __init__(self,workers=1,reuse=None):
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=64:
            raise ValueError('media_workers must be between 1 and 64')
        self.workers=workers;self.pool=None;self.reuse=Path(reuse).resolve() if reuse else None
        self.resources={}
        if self.reuse:
            from .package_layout import verify_package_provenance
            if (self.reuse/'BUILDING').exists():raise MediaReuseError('Cannot reuse unfinished media package')
            summary=json.loads((self.reuse/'dataset_metadata.json').read_text())
            verify_package_provenance(self.reuse,summary);self.resources=summary['resources']
    def __enter__(self):return self
    def __exit__(self,*args):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
    def validate(self,directory,record,eid):
        directory=Path(directory)
        references=[ref for obs in record['observations'] for ref in obs['images'].values()]
        source=directory/'trajectory.hdf5'
        before=source.stat() if source.is_file() else None
        digest=_hash(source) if before else None
        if self.reuse:
            from .package_layout import resource_path
            res=self.resources.get(eid)
            if not digest or not res or res.get('media_source_sha256')!=digest:
                raise MediaReuseError('Source media changed or lacks content-verified reuse provenance')
            path=resource_path(self.reuse,res,'media_manifest')
            if _hash(path)!=res['media_manifest_sha256']:raise MediaReuseError('Image manifest changed')
            media=json.loads(path.read_text())
            if set(media)!=set(references):raise MediaReuseError('Source image references changed')
            # Preserve the original reference insertion order, hence identical JSON bytes.
            media={ref:media[ref] for ref in references}
        else:
            jobs=[(str(directory),references[i:i+128]) for i in range(0,len(references),128)]
            if self.workers==1:results=map(validate_batch,jobs)
            else:
                if self.pool is None:
                    method='forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
                    self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context(method))
                results=self.pool.map(validate_batch,jobs)
            media={ref:sha for batch in results for ref,sha in batch}
        if before:
            after=source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise MediaReuseError('Source media changed during validation')
        return media,digest
