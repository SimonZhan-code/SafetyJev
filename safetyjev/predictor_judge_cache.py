"""Disposable indexed image/numeric cache for recorded Predictor Judge windows."""
import hashlib,io,json,os,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from .visual_cache import file_hash,_atomic_json,sample_index_digest


def window_key(row):
    fields=[row[k] for k in ('episode_id','start_step','proposal_id','action_offset','history_steps')]
    return hashlib.sha256(json.dumps(fields,separators=(',',':')).encode()).hexdigest()


def build_cache(package,output,*,max_bytes=None):
    from .predictor_judge_dataset import PredictorJudgeWindowDataset
    from .predictor_judge_schema import make_action_window,CAMERAS
    import fcntl
    package=Path(package).resolve();output=Path(output).resolve()
    if output==package:raise ValueError('Cache needs its own directory')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes<1):raise ValueError('Invalid byte budget')
    identity=file_hash(package/'dataset_metadata.json');output.mkdir(parents=True,exist_ok=True)
    with (output/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/'cache.json'
        if manifest.exists():
            previous=json.loads(manifest.read_text())
            if previous['package_sha256']!=identity:raise ValueError('Cache source identity differs')
            if previous['complete']:return validate_cache(output,package)
        _atomic_json(manifest,{'kind':'predictor_judge','complete':False,'package_sha256':identity})
        db=sqlite3.connect(output/'frames.sqlite')
        try:
            db.executescript('''CREATE TABLE IF NOT EXISTS images(key TEXT PRIMARY KEY,png BLOB,sha TEXT);
              CREATE TABLE IF NOT EXISTS windows(key TEXT PRIMARY KEY,payload TEXT,sha TEXT);
              CREATE TABLE IF NOT EXISTS samples(split TEXT,idx INTEGER,offset INTEGER,PRIMARY KEY(split,idx));''')
            def budget():
                if max_bytes is not None and sum(p.stat().st_size for p in output.iterdir() if p.is_file())>max_bytes:
                    raise ValueError('Cache byte budget exceeded; incomplete cache can resume')
            db.execute('DELETE FROM samples');db.commit();budget()
            counts={}
            for split in ['train','validation','test']:
                # Train-only review packages legitimately have no held-out rows.
                # Verify the empty file before skipping the nonempty loader.
                split_path=package/(split+'.jsonl')
                if split_path.stat().st_size==0:
                    summary=json.loads((package/'dataset_metadata.json').read_text())
                    if file_hash(split_path)!=summary['file_sha256'][split]:raise ValueError('Split bytes changed')
                    counts[split]=0
                    continue
                data=PredictorJudgeWindowDataset(package,split);counts[split]=len(data)
                for idx in range(len(data)):
                    row=data.record(idx);key=window_key(row)
                    db.execute('INSERT INTO samples VALUES (?,?,?)',(split,idx,int(data.offsets[idx])))
                    if not db.execute('SELECT 1 FROM windows WHERE key=?',(key,)).fetchone():
                        root,e,media=data._episode(row['episode_id']);t=row['start_step']
                        proposal=next(p for p in e['proposals'] if p['proposal_id']==row['proposal_id'])
                        w=make_action_window(proposal['planned_commands'],row['action_offset'],dt_s=e['action_dt_s'])
                        values={'robot_state':e['observations'][t]['robot_state'],'remaining_actions':w['actions'].tolist(),
                                'action_mask':w['action_mask'].tolist(),'action_dt_s':w['action_dt_s'],
                                'history_mask':[s is not None for s in row['history_steps']],'views':{}}
                        if row.get('input_contract')=='chunk_start_v2':
                            from .chunk_judge import chunk_numeric_inputs
                            for name,value in chunk_numeric_inputs(e,row).items():
                                values[name]=value.tolist() if isinstance(value,np.ndarray) else value
                        for camera in CAMERAS:
                            keys=[]
                            for step in row['history_steps']:
                                relative=e['observations'][t if step is None else step]['images'][camera]
                                imagekey=hashlib.sha256(json.dumps([row['episode_id'],relative]).encode()).hexdigest()
                                if not db.execute('SELECT 1 FROM images WHERE key=?',(imagekey,)).fetchone():
                                    blob=data.read_image(root,relative,media[relative]);digest=hashlib.sha256(blob).hexdigest()
                                    with Image.open(io.BytesIO(blob)) as im:im.load()
                                    db.execute('INSERT INTO images VALUES (?,?,?)',(imagekey,blob,digest))
                                keys.append(imagekey)
                            values['views'][camera]=keys
                        payload=json.dumps(values,allow_nan=False,separators=(',',':'))
                        db.execute('INSERT INTO windows VALUES (?,?,?)',(key,payload,hashlib.sha256(payload.encode()).hexdigest()))
                    if idx%100==0:db.commit();budget()
                db.commit();budget();data.close()
            for table,column in [('windows','payload'),('images','png')]:
                for payload,digest in db.execute(f'SELECT {column},sha FROM {table}'):
                    if isinstance(payload,str):payload=payload.encode()
                    if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Cache payload checksum differs')
            result={'kind':'predictor_judge','complete':True,'package_sha256':identity,'splits':counts,'samples':sum(counts.values()),
                    'images':db.execute('SELECT count(*) FROM images').fetchone()[0],'windows':db.execute('SELECT count(*) FROM windows').fetchone()[0],
                    'index_sha256':{s:sample_index_digest((i,o,0) for i,o in db.execute('SELECT idx,offset FROM samples WHERE split=? ORDER BY idx',(s,))) for s in counts},
                    'database_sha256':file_hash(output/'frames.sqlite'),'database_bytes':(output/'frames.sqlite').stat().st_size}
            budget();_atomic_json(manifest,result);return result
        finally:db.close()


class JudgeCache:
    def __init__(self,root,package):
        self.root=Path(root).resolve();self.meta=json.loads((self.root/'cache.json').read_text())
        if not self.meta.get('complete'):raise ValueError('Incomplete predictor judge cache')
        if self.meta.get('kind')!='predictor_judge' or self.meta['package_sha256']!=file_hash(Path(package)/'dataset_metadata.json'):
            raise ValueError('Cache source identity differs')
        self.db=None;self.pid=None
    def connection(self):
        if self.pid!=os.getpid():
            self.close();self.db=sqlite3.connect((self.root/'frames.sqlite').as_uri()+'?mode=ro',uri=True)
            self.db.execute('PRAGMA cache_size=-8192');self.pid=os.getpid()
        return self.db
    def close(self):
        if self.db is not None:self.db.close()
        self.db=None;self.pid=None
    def __getstate__(self):
        state=self.__dict__.copy();state.update(db=None,pid=None);return state
    def inputs(self,row):
        item=self.connection().execute('SELECT payload,sha FROM windows WHERE key=?',(window_key(row),)).fetchone()
        if item is None:raise ValueError('Missing cached window')
        payload,digest=item
        if hashlib.sha256(payload.encode()).hexdigest()!=digest:raise ValueError('Numeric window checksum differs')
        values=json.loads(payload);views={}
        for camera,keys in values.pop('views').items():
            images=[]
            for available,key in zip(values['history_mask'],keys):
                item=self.connection().execute('SELECT png,sha FROM images WHERE key=?',(key,)).fetchone()
                if item is None:raise ValueError('Missing cached image')
                blob,sha=item
                if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('Cached image checksum differs')
                with Image.open(io.BytesIO(blob)) as im:image=np.asarray(im.convert('RGB'),dtype=np.uint8).transpose(2,0,1).copy()
                images.append(image if available else np.zeros_like(image))
            views[camera]=np.stack(images)
        for key in ['robot_state','remaining_actions']:values[key]=np.asarray(values[key],dtype=np.float32)
        for key in ['action_mask','history_mask']:values[key]=np.asarray(values[key],dtype=bool)
        if 'executed_actions' in values:
            values['executed_actions']=np.asarray(values['executed_actions'],dtype=np.float32)
            values['history_robot_states']=np.asarray(values['history_robot_states'],dtype=np.float32)
            values['history_action_mask']=np.asarray(values['history_action_mask'],dtype=bool)
        values.update(questions=row['question'],observations=views,constraint_context=row['constraint_context']);return values


def validate_cache(root,package):
    reader=JudgeCache(root,package)
    if file_hash(Path(root)/'frames.sqlite')!=reader.meta['database_sha256']:raise ValueError('Cache database checksum differs')
    return reader.meta
