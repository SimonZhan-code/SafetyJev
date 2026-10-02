"""Passive raw recording; no model calls and no returned policy actions."""
import copy,json,math,time
from pathlib import Path
import numpy as np


def plain(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


class RolloutRecorder:
    def __init__(self,directory,metadata,*,chunk_size=128):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        if (self.directory/'episode.json').exists():raise FileExistsError(self.directory)
        if type(chunk_size) is not int or chunk_size<1:raise ValueError('Positive flush chunk size required')
        for name in ['observations','states','snapshots']:(self.directory/name).mkdir()
        self.record={**plain(metadata),'observations':[],'proposals':[],'execution':[],'oracle':[]}
        self.pending=None;self.applied=False;self.buffer=[];self.chunk_size=chunk_size;self.finished=False
        self._write_json('episode.json',metadata);self._write_json('episode_status.json',{'recording_status':'recording'})
    def _write_json(self,name,data):
        path=self.directory/name;tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def _append(self,name,data):
        row=plain(data)
        with (self.directory/name).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        return row
    def event(self,kind,**values):self._append('events.jsonl',{'event':kind,'monotonic_s':time.monotonic(),**values})
    def start_episode(self,initial_record,oracle):
        if self.record['observations']:raise ValueError('Episode already started')
        self._observation(initial_record);self.record_oracle(oracle)
    def record_proposal(self,proposal_record):
        row=self._append('proposals.jsonl',proposal_record);self.record['proposals'].append(row)
    def before_action(self,execution_record):
        if self.pending is not None:raise ValueError('Previous action attempt has no successor observation')
        self.pending=self._append('attempts.jsonl',execution_record);self.applied=False
    def action_applied(self,step):
        if self.pending is None or self.pending['step']!=step or self.applied:raise ValueError('Unpaired/repeated env step')
        self.record['execution'].append(self._append('execution.jsonl',self.pending));self.applied=True
        self.event('env_step_returned',step=step)
    def after_step(self,step_record):
        if self.pending is None or step_record['step']!=self.pending['step']+1:raise ValueError('Unpaired action/observation')
        if not self.applied:self.action_applied(self.pending['step'])
        self._observation(step_record)
        self.pending=None;self.applied=False
    def _observation(self,row):
        from PIL import Image
        t=row['step']
        if t!=len(self.record['observations']):raise ValueError('Observation step gap')
        images={}
        for camera,array in row['images'].items():
            if camera not in ('overview','wrist'):raise ValueError('Unexpected camera')
            file=f'observations/{t:07d}-{camera}.png';pixels=np.array(array,dtype=np.uint8,copy=True)
            Image.fromarray(pixels).save(self.directory/file);images[camera]=file
        if len(images)!=2:raise ValueError('Missing camera')
        finite=bool(np.isfinite(np.asarray(row['robot_state'],dtype=float)).all())
        metadata={'step':t,'robot_state':plain(row['robot_state']),'images':images,'valid':finite,
                  'simulation_time_s':t*self.record['action_dt_s'],'monotonic_s':time.monotonic()}
        self.record['observations'].append(self._append('observations.jsonl',metadata))
        self._append('states/physical.jsonl',{'step':t,'physical':row.get('physical',{}),'policy_state':row.get('policy_state')})
        self.buffer.append((t,np.asarray(row['robot_state'],dtype=np.float32).copy()))
        if len(self.buffer)>=self.chunk_size:self._flush()
    def record_oracle(self,row):
        if any(x['step']==row['step'] for x in self.record['oracle']):raise ValueError('Duplicate oracle step')
        self.record['oracle'].append(self._append('oracle.jsonl',row))
    def _flush(self):
        if not self.buffer:return
        name=f'states/{self.buffer[0][0]:07d}.npz';tmp=self.directory/(name+'.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,steps=np.asarray([x[0] for x in self.buffer]),robot_state=np.stack([x[1] for x in self.buffer]))
        tmp.replace(self.directory/name);self.buffer=[]
    def finish_episode(self,status_record):
        if self.finished:raise ValueError('Already finalized')
        seen={r['step'] for r in self.record['oracle']}
        for obs in self.record['observations']:
            if obs['step'] not in seen:self.record_oracle({'step':obs['step'],'valid':False,'error':'monitor sample unavailable','ap':None,'violated':{}})
        self.record['oracle'].sort(key=lambda r:r['step']);self._flush()
        complete=(status_record.get('status')=='completed' and self.pending is None
                  and all(r['valid'] for r in self.record['oracle']) and bool(self.record['observations']))
        self.record['result']=plain(status_record);self._write_json('record.json',self.record)
        self._write_json('episode_status.json',{'recording_status':'complete' if complete else 'incomplete',
             'steps':len(self.record['execution']),'observations':len(self.record['observations']),
             'pending_attempt':self.pending,'pending_env_step_returned':self.applied, 'result':status_record});self.finished=True
        self.event('finish',recording_status='complete' if complete else 'incomplete')


def create_observer(*,cfg,scene):
    from .predictor_judge_observer import ManiGuardObserver
    return ManiGuardObserver(cfg,scene)
