"""Portable predictor judge indices and a masked multimodal DataLoader."""
from array import array
from collections import OrderedDict,Counter
import hashlib,json,os,io
from pathlib import Path
import numpy as np
from .predictor_judge_schema import make_action_window,STATE_FEATURES,CAMERAS
from .predictor_judge_data import build_predictor_judge_samples,split_groups


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def build_package(episode_dirs,output,*,history_frames=3,seed=42,fractions=(.8,.1,.1),
                  group_splits=None,unsafe_per_safe=4,active_motion_rad=.05):
    from PIL import Image
    from .predictor_judge_curation import inventory_episode,select_episodes,composition_report,write_summary
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'BUILDING').touch()
    inventory=[];candidates={};resources={}
    (output/'media').mkdir()
    for directory in sorted(Path(d).resolve() for d in episode_dirs):
        try:
            e=json.loads((directory/'record.json').read_text())
        except (OSError,ValueError):
            # A crashed attempt may have episode.json but no finalized record.
            try:e=json.loads((directory/'episode.json').read_text())
            except (OSError,ValueError):e={}
            e.setdefault('episode_id',directory.name)
            e.setdefault('group_id',None)
        info=inventory_episode(e,active_motion_rad=active_motion_rad)
        eid=info['episode_id']
        if not isinstance(eid,str) or not eid or eid in candidates:raise ValueError('Missing/duplicate episode ID')
        media={}
        if info['quality_ok']:
            try:
                for observation in e['observations']:
                    for relative in observation['images'].values():
                        file=(directory/relative).resolve()
                        if not file.is_relative_to(directory):raise ValueError('Image path escapes episode')
                        media[relative]=file_hash(file)
                        with Image.open(file) as im:im.load()
            except (OSError,ValueError):
                info['quality_ok']=False;info['quality_reasons'].append('invalid_media')
        info.update(raw_root=os.path.relpath(directory,output),
                    record_sha256=file_hash(directory/'record.json') if (directory/'record.json').is_file() else None)
        inventory.append(info);candidates[eid]=(directory,media)
    groups=[r for r in inventory if r['group_id'] is not None]
    assignment=dict(group_splits) if group_splits is not None else split_groups(groups,seed=seed,fractions=fractions)
    if any(v not in ('train','validation','test') for v in assignment.values()):raise ValueError('Invalid frozen split')
    selected=select_episodes(groups,assignment,seed=seed,unsafe_per_safe=unsafe_per_safe)
    # Unidentified failed attempts stay in the inventory, never join a split.
    for r in inventory:
        if r['group_id'] is None:
            selected.append({**r,'split':None,'selected':False,'selection_reason':'unidentified_failed_attempt'})
    selected.sort(key=lambda r:r['episode_id'])
    with (output/'episode_inventory.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r,allow_nan=False)+'\n')
    sums={k:[np.zeros(d),np.zeros(d),0] for k,d in [('state',16),('action',8)]}
    counts=Counter();reasons=Counter()
    windows={key:{} for key in ['by_split','by_family','by_policy','by_checkpoint_id','by_constraint','by_valid_steps','by_label_reason']}
    writers={name:(output/(name+'.jsonl')).open('w') for name in ['train','validation','test','excluded']}
    try:
        for info in selected:
            if not info['selected']:continue
            eid=info['episode_id'];directory,media=candidates[eid]
            e=json.loads((directory/'record.json').read_text());split=info['split']
            media_file='media/'+hashlib.sha256(eid.encode()).hexdigest()+'.json'
            (output/media_file).write_text(json.dumps(media)+'\n')
            resources[eid]={'raw_root':info['raw_root'],'record_sha256':info['record_sha256'],'group_id':info['group_id'],
                            'media_manifest':media_file,'media_manifest_sha256':file_hash(output/media_file)}
            first={event['constraint_id']:event['step'] for event in info['events']}
            for row in build_predictor_judge_samples(e,history_frames=history_frames):
                row.update(split=split,family=info['family'],policy=info['policy'],checkpoint_id=info['checkpoint_id'],episode_safety=info['safety'])
                # Sampling metadata stays outside the model input allowlist.
                t=first.get(row['constraint_id'])
                row['negative_stratum']='near_event' if t is not None and 0<t-row['end_step']<=8 else 'ordinary'
                reasons[row['label_reason']]+=1;destination=split if row['target'] is not None else 'excluded'
                writers[destination].write(json.dumps(row,allow_nan=False)+'\n');counts[destination]+=1
                label='excluded' if row['target'] is None else ('positive' if row['target'][1] else 'negative')
                for field,value in [('split',split),('family',info['family']),('policy',info['policy']),('checkpoint_id',info['checkpoint_id']),
                                    ('constraint',info['family']+'/'+row['constraint_id']),('valid_steps',str(row['valid_steps'])),('label_reason',row['label_reason'])]:
                    counter=windows['by_'+field].setdefault(value,Counter());counter[label]+=1
            if split=='train':
                values={'state':[o['robot_state'] for o in e['observations']], 'action':[a['command'] for a in e['execution']]}
                for key,v in values.items():
                    if not v:continue
                    x=np.asarray(v,dtype=np.float64);sums[key][0]+=x.sum(0);sums[key][1]+=(x*x).sum(0);sums[key][2]+=len(x)
    finally:
        for writer in writers.values():writer.close()
    report=composition_report(selected,windows)
    report['selection']={'unsafe_per_safe':unsafe_per_safe,'active_motion_rad':active_motion_rad,
                         'split_source':'frozen_manifest' if group_splits is not None else 'engineering_auto_split',
                         'safe_quota':'per training family, ceil(unsafe/ratio); one active safe if no unsafe'}
    (output/'composition.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');write_summary(output/'DATA_SUMMARY.md',report)
    normalization={}
    for key,(total,square,n) in sums.items():
        if not n:raise ValueError('Selected training records cannot fit '+key+' normalization; inspect composition.json')
        mean=total/n;std=np.maximum(np.sqrt(np.maximum(square/n-mean*mean,0)),1e-6)
        normalization[key+'_mean']=mean.tolist();normalization[key+'_std']=std.tolist()
    metadata={'method':'per_step_action_conditioned_safety_judgment','history_frames':history_frames,'max_actions':8,
        'state_features':STATE_FEATURES,'resources':resources,'group_splits':assignment,'normalization':normalization,
        'selection':report['selection'],'counts':dict(counts),'label_reasons':dict(reasons),
        'inventory_sha256':file_hash(output/'episode_inventory.jsonl'),
        'file_sha256':{name:file_hash(output/(name+'.jsonl')) for name in writers}}
    (output/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(output/'BUILDING').unlink()
    return metadata


class PredictorJudgeWindowDataset:
    def __init__(self,package,split,cache_episodes=2):
        self.package=Path(package).resolve()
        if split not in ('train','validation','test'):raise ValueError('Use train/validation/test')
        if (self.package/'BUILDING').exists():raise ValueError('Unfinished package')
        self.summary=json.loads((self.package/'dataset_metadata.json').read_text());self.path=self.package/(split+'.jsonl')
        if file_hash(self.path)!=self.summary['file_sha256'][split]:raise ValueError('Split bytes changed')
        self.offsets=array('Q');self.cache=OrderedDict();self.cache_episodes=cache_episodes
        with self.path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                r=json.loads(line)
                if r['target'] not in ([1.,0.],[0.,1.]) or r['split']!=split or self.summary['group_splits'][r['group_id']]!=split:
                    raise ValueError('Invalid label/split assignment')
                self.offsets.append(offset)
        if not self.offsets:raise ValueError('Selected split has no eligible samples')
    def __len__(self):return len(self.offsets)
    def record(self,index):
        with self.path.open('rb') as f:f.seek(self.offsets[index]);return json.loads(f.readline())
    def _episode(self,eid):
        if eid not in self.cache:
            res=self.summary['resources'][eid];root=(self.package/res['raw_root']).resolve()
            if file_hash(root/'record.json')!=res['record_sha256']:raise ValueError('Raw record bytes changed')
            manifest=(self.package/res['media_manifest']).resolve()
            if not manifest.is_relative_to(self.package) or file_hash(manifest)!=res['media_manifest_sha256']:raise ValueError('Image manifest changed')
            self.cache[eid]=(root,json.loads((root/'record.json').read_text()),json.loads(manifest.read_text()))
            while len(self.cache)>self.cache_episodes:self.cache.popitem(last=False)
        self.cache.move_to_end(eid);return self.cache[eid]
    def __getitem__(self,index):
        from PIL import Image
        row=self.record(index);root,e,media=self._episode(row['episode_id']);t=row['start_step']
        proposal=next(p for p in e['proposals'] if p['proposal_id']==row['proposal_id'])
        w=make_action_window(proposal['planned_commands'],row['action_offset'],dt_s=e['action_dt_s'])
        views={}
        for camera in CAMERAS:
            frames=[]
            for step in row['history_steps']:
                chosen=t if step is None else step
                path=(root/e['observations'][chosen]['images'][camera]).resolve()
                if not path.is_relative_to(root):raise ValueError('Image path escapes episode')
                blob=path.read_bytes();expected=media[e['observations'][chosen]['images'][camera]]
                if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('Image bytes changed')
                with Image.open(io.BytesIO(blob)) as im:frame=np.asarray(im.convert('RGB'),dtype=np.uint8).transpose(2,0,1).copy()
                frames.append(np.zeros_like(frame) if step is None else frame)
            views[camera]=np.stack(frames)
        return {'inputs':{'questions':row['question'],'observations':views,'robot_state':np.asarray(e['observations'][t]['robot_state'],dtype=np.float32),
            'remaining_actions':w['actions'],'action_mask':w['action_mask'],'action_dt_s':w['action_dt_s'],
            'history_mask':np.asarray([s is not None for s in row['history_steps']]),'constraint_context':row['constraint_context']},
            'target':row['target'],'sample_id':row['id'],'query_id':row['constraint_id'],'valid_steps':w['valid_steps']}


def collate_predictor_judge(items):
    import torch
    if not items:raise ValueError('Empty batch')
    inputs={'questions':[r['inputs']['questions'] for r in items],
            'constraint_context':[r['inputs']['constraint_context'] for r in items],
            'observations':{c:torch.from_numpy(np.stack([r['inputs']['observations'][c] for r in items])) for c in CAMERAS}}
    for key in ['robot_state','remaining_actions','action_mask','action_dt_s','history_mask']:
        inputs[key]=torch.from_numpy(np.asarray([r['inputs'][key] for r in items]))
    return {'inputs':inputs,'targets':torch.tensor([r['target'] for r in items],dtype=torch.float32),
            'sample_ids':[r['sample_id'] for r in items],'query_ids':[r['query_id'] for r in items],
            'valid_steps':[r['valid_steps'] for r in items]}
