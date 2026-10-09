"""Parallel annotation preparation for an explicitly validated source cohort."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path


def _prepare(job):
    directory,eid,output,definitions,asset_root,reuse_dirs,prepare_records=job
    from .semantic_data import prepare_semantic_annotation,write_annotation
    name=hashlib.sha256(eid.encode()).hexdigest()+'.json'
    if prepare_records:_write_record(directory,eid,Path(output),name)
    reuse=next((root for root in reuse_dirs if (Path(root)/name).is_file()),None)
    annotation=prepare_semantic_annotation(directory,eid,definitions,
        liquid_asset_root=asset_root,reuse_annotations=reuse)
    target=Path(output)/name;temporary=target.with_suffix('.tmp')
    write_annotation(temporary,annotation);temporary.replace(target)
    return eid


def prepare_annotations(sources,output,definitions,*,workers=16,asset_root=None,reuse_dirs=(),prepare_records=False):
    """Yield completed IDs; final builders retain their own deterministic order.

    Sources must already have passed cohort quality checks. Reused annotations
    undergo the same implementation/source identity checks as serial preparation.
    """
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=64:
        raise ValueError('annotation workers must be between 1 and 64')
    sources=list(sources)
    if len({eid for _,eid in sources})!=len(sources):raise ValueError('Duplicate annotation episode')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if prepare_records:
        (output/'records').mkdir();(output/'receipts').mkdir()
    jobs=[(str(directory),eid,str(output),definitions,asset_root,list(map(str,reuse_dirs)),prepare_records) for directory,eid in sources]
    if workers==1:
        for job in jobs:yield _prepare(job)
    else:
        method='forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context(method)) as pool:
            futures=[pool.submit(_prepare,job) for job in jobs]
            try:
                for future in as_completed(futures):yield future.result()
            except BaseException:
                for future in futures:future.cancel()
                raise


def _identity(directory):
    from .media_preparation import _hash
    root=Path(directory)
    return {name:_hash(root/name) if (root/name).is_file() else None for name in
            ('episode.json','trajectory.hdf5','snapshots.hdf5','task_scene.json','diagnostics.jsonl','events.jsonl')}


def _implementation():
    from . import source_episodes,predictor_judge_schema,predictor_judge_observer
    from maniguard.data.recording import reader
    from .media_preparation import _hash
    return {m.__name__:_hash(m.__file__) for m in (source_episodes,predictor_judge_schema,predictor_judge_observer,reader)}


def _write_record(directory,eid,output,name):
    from .source_episodes import load_source_record
    from .media_preparation import _hash
    identity=_identity(directory);record=load_source_record(directory)
    if record['episode_id']!=eid:raise ValueError('Prepared record episode mismatch')
    if identity!=_identity(directory):raise ValueError('Source files changed during record preparation')
    target=output/'records'/name;tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,allow_nan=False)+'\n');tmp.replace(target)
    receipt=dict(source_files=identity,implementation=_implementation(),record_sha256=_hash(target))
    target=output/'receipts'/name;tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt)+'\n');tmp.replace(target)


def load_prepared_record(directory,cache):
    from .media_preparation import _hash,MediaReuseError
    directory=Path(directory);cache=Path(cache)
    eid=json.loads((directory/'episode.json').read_text())['episode_id']
    name=hashlib.sha256(eid.encode()).hexdigest()+'.json'
    receipt=json.loads((cache/'receipts'/name).read_text())
    if receipt['implementation']!=_implementation():raise MediaReuseError('Record adapter implementation changed')
    if receipt['source_files']!=_identity(directory):raise MediaReuseError('Source files changed since record preparation')
    path=cache/'records'/name
    if _hash(path)!=receipt['record_sha256']:raise MediaReuseError('Prepared record changed')
    record=json.loads(path.read_text())
    if record['episode_id']!=eid:raise MediaReuseError('Prepared record episode mismatch')
    return record
