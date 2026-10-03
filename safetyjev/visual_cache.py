"""Rebuildable, lossless current-frame cache with transactional preparation."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import struct

import numpy as np
from PIL import Image


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def frame_key(resource, video, frame, fps):
    return hashlib.sha256(json.dumps([resource,video,int(frame),float(fps)],separators=(',',':')).encode()).hexdigest()


def sample_index_digest(rows):
    digest=hashlib.sha256()
    for idx,offset,stratum in rows:digest.update(struct.pack('!QQI',idx,offset,stratum))
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def build_visual_cache(package, output, *, max_bytes=None):
    """Prepare all indexed splits. A byte-limited failure stays resumable, not complete."""
    from .visual_train import verify_package
    package=Path(package).resolve();output=Path(output).resolve()
    if output==package:raise ValueError('Cache must have its own directory')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes<1):raise ValueError('Invalid byte budget')
    summary=verify_package(package)
    identity=file_hash(package/'dataset_metadata.json')
    output.mkdir(parents=True,exist_ok=True)
    # An OS lock prevents concurrent writers; readers only admit completed caches.
    import fcntl
    with (output/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/'cache.json'
        if manifest.exists():
            previous=json.loads(manifest.read_text())
            if previous['package_sha256']!=identity:raise ValueError('Cache source identity differs')
            if previous['complete']:
                validate_cache(output,package)
                return previous
        _atomic_json(manifest,{'format':1,'complete':False,'package_sha256':identity})
        db=sqlite3.connect(output/'frames.sqlite')
        try:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA cache_size=-8192')
            db.executescript('''
              CREATE TABLE IF NOT EXISTS frames (key TEXT PRIMARY KEY, resource TEXT, video TEXT, frame INTEGER, fps REAL, png BLOB, sha TEXT);
              CREATE INDEX IF NOT EXISTS video_frames ON frames(resource,video,fps,frame);
              CREATE TABLE IF NOT EXISTS samples (split TEXT, idx INTEGER, offset INTEGER, stratum INTEGER, PRIMARY KEY(split,idx));
              CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value TEXT);
            ''')
            def budget():
                size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                if max_bytes is not None and size>max_bytes:raise ValueError('Cache byte budget exceeded; incomplete cache retained for resume')
            if not db.execute("SELECT 1 FROM progress WHERE key='indexed'").fetchone():
                # Indexing is idempotent after interruption; frame payloads are filled only afterwards.
                db.execute('DELETE FROM samples');db.execute('DELETE FROM frames');db.commit()
                for split in summary['file_sha256']:
                    strata={};idx=0
                    with (package/(split+'.jsonl')).open('rb') as stream:
                        while True:
                            offset=stream.tell();line=stream.readline()
                            if not line:break
                            if not line.strip():continue
                            row=json.loads(line);media=row['state']['observation_window']
                            if len(media['frame_indices'])!=1:raise ValueError('Classifier cache requires one current frame')
                            if row['target'] not in ([1,0],[0,1]) or row['options']!=['no','yes']:raise ValueError('Expected binary Noul sample')
                            if row.get('split',split)!=split:raise ValueError('Sample split mismatch')
                            group=(row['metadata']['family'],row['metadata']['query_id'],int(row['target'][1]))
                            code=strata.setdefault(group,len(strata))
                            db.execute('INSERT INTO samples VALUES (?,?,?,?)',(split,idx,offset,code))
                            frame=media['frame_indices'][0];fps=media['video_fps']
                            if type(frame) is not int or frame<0 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid frame clock')
                            for video in media['videos'].values():
                                resource=media['resource_id'];key=frame_key(resource,video,frame,fps)
                                db.execute('INSERT OR IGNORE INTO frames(key,resource,video,frame,fps) VALUES (?,?,?,?,?)',(key,resource,video,frame,fps))
                            idx+=1
                            if idx%10000==0:db.commit();budget()
                db.execute("INSERT OR REPLACE INTO progress VALUES ('indexed','true')");db.commit()
            budget()
            videos=db.execute('SELECT DISTINCT resource,video,fps FROM frames WHERE png IS NULL').fetchall()
            import av
            for resource,video,fps in videos:
                raw=(package/summary['resources'][resource]['raw_root']).resolve();path=(raw/video).resolve()
                if not path.is_relative_to(raw):raise ValueError('Video escapes source root')
                wanted=dict(db.execute('SELECT frame,key FROM frames WHERE resource=? AND video=? AND fps=? AND png IS NULL',(resource,video,fps)))
                with av.open(str(path)) as container:
                    stream=container.streams.video[0];stream.codec_context.thread_count=1
                    if not math.isclose(float(stream.average_rate),fps,abs_tol=1e-6):raise ValueError('Video rate differs from index')
                    start=stream.start_time or 0
                    for frame in container.decode(stream):
                        if frame.pts is None:raise ValueError('Video frame lacks timestamp')
                        position=float((frame.pts-start)*stream.time_base)*fps;index=round(position)
                        if abs(position-index)>.01:raise ValueError('Video frame clock is not aligned')
                        if index not in wanted:continue
                        pixels=frame.to_ndarray(format='rgb24');buffer=io.BytesIO()
                        Image.fromarray(pixels).save(buffer,format='PNG',compress_level=1);payload=buffer.getvalue()
                        db.execute('UPDATE frames SET png=?,sha=? WHERE key=?',(payload,hashlib.sha256(payload).hexdigest(),wanted.pop(index)))
                        # Commit bounded groups, so interruption does not discard entire videos.
                        if index%100==0:db.commit();budget()
                        if not wanted:break
                if wanted:raise ValueError('Missing requested frames: '+str(path))
                db.commit();budget()
            counts=dict(db.execute('SELECT split,count(*) FROM samples GROUP BY split'))
            report={'format':1,'complete':True,'package_sha256':identity,'frames':db.execute('SELECT count(*) FROM frames').fetchone()[0],
                    'samples':sum(counts.values()),'splits':counts,'database_bytes':(output/'frames.sqlite').stat().st_size}
            if db.execute('SELECT count(*) FROM frames WHERE png IS NULL').fetchone()[0]:raise ValueError('Cache is incomplete')
            # Full integrity check also verifies payloads from earlier interrupted invocations.
            for payload,digest in db.execute('SELECT png,sha FROM frames'):
                if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Cached image checksum mismatch')
            report['sample_index_sha256']={split:sample_index_digest(db.execute('SELECT idx,offset,stratum FROM samples WHERE split=? ORDER BY idx',(split,))) for split in counts}
            report['database_sha256']=file_hash(output/'frames.sqlite')
            budget();_atomic_json(manifest,report)
            return report
        finally:db.close()


class FrameCache:
    def __init__(self, root, package):
        self.root=Path(root).resolve();self.package=Path(package).resolve()
        meta=json.loads((self.root/'cache.json').read_text())
        if not meta.get('complete'):raise ValueError('Frame cache is not complete')
        if meta['package_sha256']!=file_hash(self.package/'dataset_metadata.json'):raise ValueError('Cache source identity differs')
        self.metadata=meta
        self.db=None;self.pid=None

    def connection(self):
        if self.pid!=os.getpid():
            self.close();self.db=sqlite3.connect((self.root/'frames.sqlite').as_uri()+'?mode=ro',uri=True)
            self.db.execute('PRAGMA cache_size=-8192');self.pid=os.getpid()
        return self.db

    def read(self, key):
        row=self.connection().execute('SELECT png,sha FROM frames WHERE key=?',(key,)).fetchone()
        if row is None:raise ValueError('Frame missing from cache')
        payload,digest=row
        if payload is None or hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Cached image checksum mismatch')
        with Image.open(io.BytesIO(payload)) as image:return np.array(image.convert('RGB'))

    def close(self):
        if self.db is not None:self.db.close()
        self.db=None;self.pid=None

    def __getstate__(self):
        state=self.__dict__.copy();state['db']=None;state['pid']=None;return state


def validate_cache(root, package):
    reader=FrameCache(root,package)
    try:
        if file_hash(Path(root)/'frames.sqlite')!=reader.metadata['database_sha256']:raise ValueError('Cache database checksum mismatch')
        db=reader.connection()
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Cache database corruption')
        for key, in db.execute('SELECT key FROM frames'):reader.read(key)
        return json.loads((Path(root)/'cache.json').read_text())
    finally:reader.close()
