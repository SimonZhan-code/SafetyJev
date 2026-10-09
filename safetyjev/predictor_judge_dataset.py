"""Portable predictor judge indices and a masked multimodal DataLoader."""
from array import array
from collections import OrderedDict,Counter
import hashlib,json,os,io
from pathlib import Path
import numpy as np
from .source_episodes import MediaReader,load_source_record
from .package_layout import split_file,resource_path,verify_package_provenance
from .predictor_judge_schema import make_action_window,STATE_FEATURES,CAMERAS
from .predictor_judge_data import build_predictor_judge_samples,split_groups


def file_hash(path):
    h=hashlib.sha256()
    with (path if hasattr(path,'open') else Path(path)).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def resource_record_path(package,res):
    package=Path(package).resolve()
    path=resource_path(package,res,'record_file') if res.get('record_file') else package/res['raw_root']/'record.json'
    return path


def verify_semantic_resource(package,res):
    if res.get('annotation_file'):
        path=resource_path(package,res,'annotation_file')
        if file_hash(path)!=res['annotation_file_sha256']:
            raise ValueError('Annotation file changed')


def build_package(episode_dirs,output,*,media_workers=1,reuse_media=None,**options):
    from .media_preparation import MediaValidator
    with MediaValidator(media_workers,reuse_media) as validator:
        return _build_package(episode_dirs,output,media_validator=validator,**options)


def _build_package(episode_dirs,output,*,media_validator,record_cache=None,history_frames=3,seed=42,fractions=(.8,.1,.1),
                  group_splits=None,unsafe_per_safe=4,active_motion_rad=.05,train_episodes='unsafe_only',
                  semantic_definitions=None,semantic_task='predictor_judge',review=False,reuse_annotations=None,liquid_asset_root=None,
                  input_contract='per_step_v1',source_provenance=None):
    from .media_preparation import MediaReuseError
    from .predictor_judge_curation import inventory_episode,select_episodes,composition_report,write_summary
    if input_contract not in ('per_step_v1','chunk_start_v2'):raise ValueError('Unsupported input contract')
    if input_contract=='chunk_start_v2':
        if semantic_definitions is None or semantic_task!='predictor_judge':raise ValueError('Chunk inputs require semantic predictor judge')
        history_frames=8
    if semantic_definitions is None and (semantic_task!='predictor_judge' or review or reuse_annotations is not None or liquid_asset_root is not None):
        raise ValueError('Semantic task/review options require semantic definitions')
    if semantic_definitions is not None:
        from .semantic_labels import validate_semantic_definitions
        from .semantic_data import build_semantic_samples,write_annotation,read_annotation,to_classifier_record,prepare_semantic_annotation
        validate_semantic_definitions(semantic_definitions,for_production=not review)
        if semantic_task not in ('classifier','predictor_judge'):raise ValueError('Invalid semantic task')
        if group_splits is None:raise ValueError('Semantic preparation requires a frozen task-group split')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'BUILDING').touch()
    inventory=[];candidates={};resources={};annotations={}
    liquid_resolver=None
    asset_root=liquid_asset_root or os.environ.get('OMNIGIBSON_DATA_PATH')
    if semantic_definitions is not None and asset_root and reuse_annotations is None:
        from .liquid_geometry import LiquidAssetResolver
        liquid_resolver=LiquidAssetResolver(asset_root)
    (output/'media').mkdir();(output/'records').mkdir()
    if semantic_definitions is not None:
        (output/'annotations').mkdir()
        (output/'semantic_definitions.json').write_text(json.dumps(semantic_definitions,indent=2)+'\n')
    for directory in sorted(Path(d).resolve() for d in episode_dirs):
        source=(directory/'trajectory.hdf5').is_file();preparation_error=None
        try:
            if source and record_cache is not None:
                from .annotation_preparation import load_prepared_record
                e=load_prepared_record(directory,record_cache)
            else:e=load_source_record(directory) if source else json.loads((directory/'record.json').read_text())
        except MediaReuseError:
            raise
        except (OSError,ValueError,KeyError) as exc:
            preparation_error=str(exc)
            # A crashed attempt may have episode.json but no finalized record.
            try:e=json.loads((directory/'episode.json').read_text())
            except (OSError,ValueError):e={}
            if isinstance(e.get('observations'),int):e={}
            e.setdefault('episode_id',directory.name)
            e.setdefault('group_id',None)
        info=inventory_episode(e,active_motion_rad=active_motion_rad)
        if preparation_error is not None:
            info['quality_ok']=False;info['quality_reasons'].append('source_preparation: '+preparation_error)
        eid=info['episode_id']
        if not isinstance(eid,str) or not eid or eid in candidates:raise ValueError('Missing/duplicate episode ID')
        if semantic_definitions is not None and info['quality_ok']:
            if not source:raise ValueError('Semantic labels require source HDF5 physical evidence')
            annotation=prepare_semantic_annotation(directory,eid,semantic_definitions,review=review,reuse_annotations=reuse_annotations,liquid_asset_root=liquid_asset_root,liquid_resolver=liquid_resolver)
            rows=build_semantic_samples(e,annotation,semantic_definitions,task=semantic_task,history_frames=history_frames,review=review,input_contract=input_contract)
            info['legacy_safety']=info['safety']
            state_values=[s['label'] for q in annotation['queries'].values() if q['temporal_kind']=='state' for s in q['states']]
            interval_rows=[r for r in rows if annotation['queries'][r['constraint_id']]['temporal_kind']=='interval']
            unavailable_states=sum(v is None for v in state_values)
            unavailable_intervals=sum(r['target'] is None for r in interval_rows)
            info['semantic_coverage']={'state_observations':len(state_values),'unavailable_state_observations':unavailable_states,
                                      'interval_queries':len(interval_rows),'unavailable_interval_queries':unavailable_intervals}
            positive=1 in state_values or any(r['target']==[0.,1.] for r in rows)
            eligible=any(r['target'] is not None for r in rows)
            info['safety']='unsafe' if positive else ('safe' if eligible and not unavailable_states and not unavailable_intervals else 'unknown')
            info['events']=[{'constraint_id':qid,'step':next(s['step'] for s in q['states'] if s['label']==1)}
                            for qid,q in annotation['queries'].items() if any(s['label']==1 for s in q['states'])]
            if not eligible:
                info['quality_ok']=False;info['quality_reasons'].append('no_eligible_semantic_samples')
            annotation_file='annotations/'+hashlib.sha256(eid.encode()).hexdigest()+'.json'
            if reuse_annotations is not None:
                # Immutable annotations share storage on the same filesystem;
                # each package retains its own portable directory entry.
                import errno,shutil
                original=Path(reuse_annotations)/Path(annotation_file).name
                try:os.link(original,output/annotation_file)
                except OSError as exc:
                    if exc.errno!=errno.EXDEV:raise
                    shutil.copyfile(original,output/annotation_file)
            else:write_annotation(output/annotation_file,annotation)
            annotations[eid]={'annotation_file':annotation_file,'annotation_file_sha256':file_hash(output/annotation_file)}
        media={};media_source_sha256=None
        if info['quality_ok']:
            try:
                media,media_source_sha256=media_validator.validate(directory,e,eid)
            except MediaReuseError:
                raise
            except (OSError,ValueError):
                info['quality_ok']=False;info['quality_reasons'].append('invalid_media')
        info.update(raw_root=os.path.relpath(directory,output),
                    record_sha256=file_hash(directory/'record.json') if (directory/'record.json').is_file() else None)
        record_file=None
        if source and info['quality_ok']:
            record_file='records/'+hashlib.sha256(eid.encode()).hexdigest()+'.json'
            (output/record_file).write_text(json.dumps(e,allow_nan=False)+'\n')
            info['record_sha256']=file_hash(output/record_file)
        inventory.append(info);candidates[eid]=(directory,media,record_file,media_source_sha256)
    groups=[r for r in inventory if r['group_id'] is not None]
    assignment=dict(group_splits) if group_splits is not None else split_groups(groups,seed=seed,fractions=fractions)
    if any(v not in ('train','validation','test') for v in assignment.values()):raise ValueError('Invalid frozen split')
    selected=select_episodes(groups,assignment,seed=seed,unsafe_per_safe=unsafe_per_safe,train_episodes=train_episodes)
    # Unidentified failed attempts stay in the inventory, never join a split.
    for r in inventory:
        if r['group_id'] is None:
            selected.append({**r,'split':None,'selected':False,'selection_reason':'unidentified_failed_attempt'})
    selected.sort(key=lambda r:r['episode_id'])
    with (output/'episode_inventory.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r,allow_nan=False)+'\n')
    sums={k:[np.zeros(d),np.zeros(d),0] for k,d in [('state',16),('action',8)]}
    counts=Counter();reasons=Counter()
    semantic_windows={};query_windows={};semantic_segments=[]
    windows={key:{} for key in ['by_split','by_family','by_policy','by_checkpoint_id','by_constraint','by_valid_steps','by_label_reason']}
    writers={name:(output/(name+'.jsonl')).open('w') for name in ['train','validation','test','excluded']}
    try:
        for info in selected:
            if not info['selected']:continue
            eid=info['episode_id'];directory,media,record_file,media_source_sha256=candidates[eid]
            e=json.loads(((output/record_file) if record_file else (directory/'record.json')).read_text());split=info['split']
            media_file='media/'+hashlib.sha256(eid.encode()).hexdigest()+'.json'
            (output/media_file).write_text(json.dumps(media)+'\n')
            resources[eid]={'raw_root':info['raw_root'],'record_sha256':info['record_sha256'],'group_id':info['group_id'],
                            'media_manifest':media_file,'media_manifest_sha256':file_hash(output/media_file)}
            if media_source_sha256:resources[eid]['media_source_sha256']=media_source_sha256
            if record_file:resources[eid]['record_file']=record_file
            if eid in annotations:resources[eid].update(annotations[eid])
            first={event['constraint_id']:event['step'] for event in info['events']}
            if semantic_definitions is not None:
                annotation=read_annotation(output/annotations[eid]['annotation_file'])
                from .semantic_reporting import state_segments
                semantic_segments.extend(dict(split=split,family=info['family'],**r) for r in state_segments(annotation))
                rows=build_semantic_samples(e,annotation,semantic_definitions,task=semantic_task,history_frames=history_frames,review=review,input_contract=input_contract)
            else:rows=build_predictor_judge_samples(e,history_frames=history_frames)
            for row in rows:
                row.update(split=split,family=info['family'],policy=info['policy'],checkpoint_id=info['checkpoint_id'],episode_safety=info['safety'])
                # Sampling metadata stays outside the model input allowlist.
                t=first.get(row['constraint_id'])
                row['negative_stratum']='near_event' if t is not None and 0<t-row['end_step']<=8 else 'ordinary'
                reasons[row['label_reason']]+=1;destination=split if row['target'] is not None else 'excluded'
                exported=to_classifier_record(row,e) if semantic_definitions is not None and semantic_task=='classifier' else row
                writers[destination].write(json.dumps(exported,allow_nan=False)+'\n');counts[destination]+=1
                label='excluded' if row['target'] is None else ('positive' if row['target'][1] else 'negative')
                if semantic_definitions is not None:
                    key=(split,info['family'],row['semantic_id'])
                    stats=semantic_windows.setdefault(key,dict(episodes=set(),positive=0,negative=0,excluded=0,excluded_reasons=Counter()))
                    stats['episodes'].add(eid);stats[label]+=1
                    if label=='excluded':stats['excluded_reasons'][row['label_reason']]+=1
                    qkey=(split,info['family'],row['constraint_id'])
                    qstats=query_windows.setdefault(qkey,dict(episodes=set(),positive=0,negative=0,excluded=0,
                        excluded_reasons=Counter(),by_starting_status={}))
                    qstats['episodes'].add(eid);qstats[label]+=1
                    if label=='excluded':qstats['excluded_reasons'][row['label_reason']]+=1
                    status={True:'already_violated',False:'not_violated',None:'unknown'}[row['starts_violated']]
                    qstats['by_starting_status'].setdefault(status,Counter())[label]+=1
                for field,value in [('split',split),('family',info['family']),('policy',info['policy']),('checkpoint_id',info['checkpoint_id']),
                                    ('constraint',info['family']+'/'+row['constraint_id']),('valid_steps',str(row.get('valid_steps',0))),('label_reason',row['label_reason'])]:
                    counter=windows['by_'+field].setdefault(value,Counter());counter[label]+=1
            if split=='train':
                values={'state':[o['robot_state'] for o in e['observations']], 'action':[a['command'] for a in e['execution']]}
                for key,v in values.items():
                    if not v:continue
                    x=np.asarray(v,dtype=np.float64);sums[key][0]+=x.sum(0);sums[key][1]+=(x*x).sum(0);sums[key][2]+=len(x)
    finally:
        for writer in writers.values():writer.close()
    report=composition_report(selected,windows)
    if semantic_definitions is not None:
        report['events']['definition']='first observed positive state per episode/semantic query; not original LTL rejections or a count of all physical events; interval-only labels excluded'
        report['supervision']='semantic_safety'
        report['state_segments']=semantic_segments
        report['state_segment_scope']='contiguous observed positive state segments per episode/query; safe-to-unsafe onsets counted separately from initial/unknown-boundary segments; correlated queries can describe the same incident; liquid intervals do not identify unique physical events'
        report['semantic_windows']=[dict(split=split,family=family,semantic_id=semantic,
            episodes=len(stats['episodes']),positive=stats['positive'],negative=stats['negative'],
            excluded=stats['excluded'],eligible=stats['positive']+stats['negative'],
            candidates=stats['positive']+stats['negative']+stats['excluded'],
            excluded_reasons=dict(stats['excluded_reasons']))
            for (split,family,semantic),stats in sorted(semantic_windows.items())]
        report['query_windows']=[dict(split=split,family=family,query_id=query,episodes=len(stats['episodes']),
            positive=stats['positive'],negative=stats['negative'],excluded=stats['excluded'],
            eligible=stats['positive']+stats['negative'],candidates=stats['positive']+stats['negative']+stats['excluded'],
            excluded_reasons=dict(stats['excluded_reasons']),by_starting_status=stats['by_starting_status'])
            for (split,family,query),stats in sorted(query_windows.items())]
    report['selection']={'train_episodes':train_episodes,'unsafe_per_safe':unsafe_per_safe,'active_motion_rad':active_motion_rad,
                         'split_source':'frozen_manifest' if group_splits is not None else 'engineering_auto_split',
                         'safe_quota':'none' if train_episodes=='unsafe_only' else 'per training family, ceil(unsafe/ratio); one active safe if no unsafe'}
    (output/'composition.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');write_summary(output/'DATA_SUMMARY.md',report)
    normalization={}
    for key,(total,square,n) in sums.items():
        if not n:raise ValueError('Selected training records cannot fit '+key+' normalization; inspect composition.json')
        mean=total/n;std=np.maximum(np.sqrt(np.maximum(square/n-mean*mean,0)),1e-6)
        normalization[key+'_mean']=mean.tolist();normalization[key+'_std']=std.tolist()
    metadata={'method':'per_step_action_conditioned_safety_judgment','history_frames':history_frames,'max_actions':8,
        'state_features':STATE_FEATURES,'resources':resources,'group_splits':assignment,'normalization':normalization,
        'normalization_statistics':{key:dict(sum=total.tolist(),sum_squares=square.tolist(),count=n)
                                    for key,(total,square,n) in sums.items()},
        'selection':report['selection'],'counts':dict(counts),'label_reasons':dict(reasons),
        'inventory_sha256':file_hash(output/'episode_inventory.jsonl'),
        'file_sha256':{name:file_hash(output/(name+'.jsonl')) for name in writers}}
    if semantic_definitions is not None:
        metadata.update(method='semantic_safety',task=semantic_task,status='review_only' if review else 'reviewed',
                        input_contract=input_contract,
                        history_frames=1 if semantic_task=='classifier' else history_frames,
                        semantic_definitions_sha256=file_hash(output/'semantic_definitions.json'))
        if semantic_task=='classifier':metadata['window']={'history_frames':1,'frame_stride':1,'sample_stride':1}
    if source_provenance is not None:
        provenance_path=output/'source_provenance.json'
        provenance_path.write_text(json.dumps(source_provenance,indent=2)+'\n')
        metadata['source_provenance_sha256']=file_hash(provenance_path)
        origins={r['episode_id']:r for r in source_provenance['selected']}
        for eid,res in resources.items():
            res['source_node']=str(origins[eid]['source_node'])
    (output/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(output/'BUILDING').unlink()
    return metadata


class PredictorJudgeWindowDataset:
    def __init__(self,package,split,cache_episodes=2,frame_cache=None):
        self.package=Path(package).resolve()
        if split not in ('train','validation','test'):raise ValueError('Use train/validation/test')
        self.split=split
        if (self.package/'BUILDING').exists():raise ValueError('Unfinished package')
        self.summary=json.loads((self.package/'dataset_metadata.json').read_text());self.path=split_file(self.package,split,self.summary)
        verify_package_provenance(self.package,self.summary)
        if self.summary.get('input_contract','per_step_v1') not in ('per_step_v1','chunk_start_v2'):
            raise ValueError('Unsupported input contract; rebuild older chunk packages for historical robot state')
        if self.summary.get('method')=='semantic_safety':
            if self.summary.get('task')!='predictor_judge':raise ValueError('Expected predictor judge package')
            if file_hash(self.package/'semantic_definitions.json')!=self.summary['semantic_definitions_sha256']:
                raise ValueError('Semantic definitions changed')
        if file_hash(self.path)!=self.summary['file_sha256'][split]:raise ValueError('Split bytes changed')
        self.media_readers=OrderedDict();self.offsets=array('Q');self.cache=OrderedDict();self.cache_episodes=cache_episodes
        self.checked_annotations=set()
        self._reader=None;self._reader_pid=None;self.frame_cache=None
        if frame_cache:
            from .predictor_judge_cache import JudgeCache
            from .visual_cache import sample_index_digest
            self.frame_cache=JudgeCache(frame_cache,self.package)
            for offset, in self.frame_cache.connection().execute('SELECT offset FROM samples WHERE split=? ORDER BY idx',(split,)):self.offsets.append(offset)
            self.frame_cache.close()
            digest=sample_index_digest((i,o,0) for i,o in enumerate(self.offsets))
            if len(self.offsets)!=self.frame_cache.meta['splits'][split] or digest!=self.frame_cache.meta['index_sha256'][split]:raise ValueError('Cached sample index checksum differs')
            if not self.offsets:raise ValueError('Selected split has no eligible samples')
            return
        with self.path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                r=json.loads(line)
                if r['target'] not in ([1.,0.],[0.,1.]) or r['split']!=split or self.summary['group_splits'][r['group_id']]!=split:
                    raise ValueError('Invalid label/split assignment')
                self.offsets.append(offset)
        if not self.offsets:raise ValueError('Selected split has no eligible samples')
    def close(self):
        for reader in getattr(self,'media_readers',{}).values():reader.close()
        self.media_readers=OrderedDict()
        self._reader_pid=None
        reader=getattr(self,'_reader',None)
        if reader is not None:reader.close();self._reader=None
        cache=getattr(self,'frame_cache',None)
        if cache is not None:cache.close()
    def __del__(self):
        self.close()
    def __len__(self):return len(self.offsets)
    def record(self,index):
        if self._reader_pid!=os.getpid():
            if self._reader is not None:self._reader.close()
            self._reader=self.path.open('rb');self._reader_pid=os.getpid()
        self._reader.seek(self.offsets[index]);return json.loads(self._reader.readline())
    def __getstate__(self):
        state=self.__dict__.copy();state.update(_reader=None,_reader_pid=None,cache=OrderedDict(),media_readers=OrderedDict(),checked_annotations=set());return state
    def _episode(self,eid):
        if eid not in self.cache:
            res=self.summary['resources'][eid];root=(self.package/res['raw_root']).resolve()
            verify_semantic_resource(self.package,res)
            record_path=resource_record_path(self.package,res)
            if file_hash(record_path)!=res['record_sha256']:raise ValueError('Raw record bytes changed')
            manifest=resource_path(self.package,res,'media_manifest')
            if file_hash(manifest)!=res['media_manifest_sha256']:raise ValueError('Image manifest changed')
            self.cache[eid]=(root,json.loads(record_path.read_text()),json.loads(manifest.read_text()))
            while len(self.cache)>self.cache_episodes:self.cache.popitem(last=False)
        self.cache.move_to_end(eid);return self.cache[eid]
    def read_image(self,root,relative,digest):
        if root not in self.media_readers:
            self.media_readers[root]=MediaReader(root)
            while len(self.media_readers)>self.cache_episodes:
                self.media_readers.popitem(last=False)[1].close()
        self.media_readers.move_to_end(root)
        return self.media_readers[root].read(relative,digest)
    def __getitem__(self,index):
        return self.sample(self.record(index))
    def sample(self,row):
        from PIL import Image
        if row.get('input_contract','per_step_v1')!=self.summary.get('input_contract','per_step_v1'):
            raise ValueError('Sample/package input contracts differ')
        eid=row['episode_id']
        if eid not in self.checked_annotations and self.summary.get('method')=='semantic_safety':
            verify_semantic_resource(self.package,self.summary['resources'][eid])
            self.checked_annotations.add(eid)
        if self.frame_cache:
            inputs=self.frame_cache.inputs(row)
            return {'inputs':inputs,'target':row['target'],'sample_id':row['id'],'query_id':row['constraint_id'],'valid_steps':int(inputs['action_mask'].sum()),'metadata':_evaluation_metadata(row)}
        root,e,media=self._episode(row['episode_id']);t=row['start_step']
        proposal=next(p for p in e['proposals'] if p['proposal_id']==row['proposal_id'])
        w=make_action_window(proposal['planned_commands'],row['action_offset'],dt_s=e['action_dt_s'])
        views={}
        for camera in CAMERAS:
            frames=[]
            for step in row['history_steps']:
                chosen=t if step is None else step
                relative=e['observations'][chosen]['images'][camera]
                blob=self.read_image(root,relative,media[relative])
                with Image.open(io.BytesIO(blob)) as im:frame=np.asarray(im.convert('RGB'),dtype=np.uint8).transpose(2,0,1).copy()
                frames.append(np.zeros_like(frame) if step is None else frame)
            views[camera]=np.stack(frames)
        numeric={}
        if row.get('input_contract')=='chunk_start_v2':
            from .chunk_judge import chunk_numeric_inputs
            numeric=chunk_numeric_inputs(e,row)
        return {'inputs':{'questions':row['question'],'observations':views,'robot_state':np.asarray(e['observations'][t]['robot_state'],dtype=np.float32),
            'remaining_actions':w['actions'],'action_mask':w['action_mask'],'action_dt_s':w['action_dt_s'],
            'history_mask':np.asarray([s is not None for s in row['history_steps']]),'constraint_context':row['constraint_context'],**numeric},
            'target':row['target'],'sample_id':row['id'],'query_id':row['constraint_id'],'valid_steps':w['valid_steps'],'metadata':_evaluation_metadata(row)}


def _evaluation_metadata(row):
    return {k:row.get(k) for k in ['episode_id','family','group_id','episode_safety','start_step','end_step','first_violation_step',
                                  'semantic_id','starts_violated','definition_sha256','annotation_sha256','input_contract','proposal_id','observed_end']}


def collate_predictor_judge(items):
    import torch
    if not items:raise ValueError('Empty batch')
    inputs={'questions':[r['inputs']['questions'] for r in items],
            'constraint_context':[r['inputs']['constraint_context'] for r in items],
            'observations':{c:torch.from_numpy(np.stack([r['inputs']['observations'][c] for r in items])) for c in CAMERAS}}
    for key in ['robot_state','remaining_actions','action_mask','action_dt_s','history_mask']:
        inputs[key]=torch.from_numpy(np.asarray([r['inputs'][key] for r in items]))
    chunk=['executed_actions' in r['inputs'] for r in items]
    if any(chunk):
        if not all(chunk):raise ValueError('Cannot mix predictor input contracts in a batch')
        for key in ('executed_actions','history_robot_states','history_action_mask'):
            inputs[key]=torch.from_numpy(np.asarray([r['inputs'][key] for r in items]))
    return {'inputs':inputs,'targets':torch.tensor([r['target'] for r in items],dtype=torch.float32),
            'sample_ids':[r['sample_id'] for r in items],'query_ids':[r['query_id'] for r in items],
            'valid_steps':[r['valid_steps'] for r in items],'metadata':[r.get('metadata',{}) for r in items]}


class PreparedPredictorJudgeCollator:
    """Worker-local CPU image/text processing; numeric tensors stay differentiable in the model."""
    def __init__(self,model_config):self.config=dict(model_config);self.processor=None
    def __call__(self,items):
        from transformers import AutoProcessor
        from jev.predictor_judge_model import encode_predictor_judge_visual_inputs
        chunk=self.config.get('input_contract')=='chunk_start_v2'
        if self.processor is None:
            os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
            self.processor=AutoProcessor.from_pretrained(self.config['model_id'],revision=self.config['revision'],
                min_pixels=self.config['min_pixels'],max_pixels=self.config['max_pixels'],local_files_only=True)
            self.processor.tokenizer.padding_side='right'
            if self.processor.tokenizer.pad_token_id is None:self.processor.tokenizer.pad_token=self.processor.tokenizer.eos_token
        batch=collate_predictor_judge(items);inputs=batch['inputs']
        visual={k:inputs[k] for k in ['questions','observations','action_mask','action_dt_s','history_mask','constraint_context']}
        if chunk:
            from .chunk_judge_model import encode_chunk_inputs
            visual['history_action_mask']=inputs['history_action_mask']
            encoded=encode_chunk_inputs(self.processor,self.config,**visual)
        else:encoded=encode_predictor_judge_visual_inputs(self.processor,self.config,**visual)
        batch['inputs']={k:inputs[k] for k in ['robot_state','remaining_actions','action_mask','action_dt_s','history_mask']}
        if chunk:
            batch['inputs'].update({k:inputs[k] for k in ('executed_actions','history_robot_states','history_action_mask')})
        batch['inputs']['encoded']=encoded;return batch
