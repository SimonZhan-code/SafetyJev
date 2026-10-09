"""Shared resources and seekable split shards for a relocatable delivery root."""
import bisect
from collections import Counter
import copy
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace


def resource_path(package, resource, field):
    package=Path(package).resolve()
    root_ref=resource.get('dataset_root','.')
    if root_ref not in ('.','..','../..','../../..'):
        raise ValueError('Invalid shared dataset root')
    root=(package/root_ref).resolve()
    relative=Path(resource[field])
    path=(package/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('Resource escapes dataset root')
    return path


class JoinedReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__();self.paths=paths;self.ends=[];total=0
        for p in paths:total+=p.stat().st_size;self.ends.append(total)
        self.position=0;self.stream=None;self.current=None
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        position=offset+(self.position if whence==1 else self.ends[-1] if whence==2 else 0)
        if whence not in (0,1,2) or position<0:raise ValueError('Invalid seek')
        self.position=position;return position
    def _stream(self):
        i=bisect.bisect_right(self.ends,self.position)
        if i==len(self.paths):return None
        if i!=self.current:
            if self.stream:self.stream.close()
            self.stream=self.paths[i].open('rb');self.current=i
        self.stream.seek(self.position-(self.ends[i-1] if i else 0))
        return self.stream
    def read(self, size=-1):
        self._checkClosed();chunks=[];remaining=size
        while remaining!=0:
            stream=self._stream()
            if stream is None:break
            data=stream.read(remaining);self.position+=len(data);chunks.append(data)
            if remaining>0:remaining-=len(data)
        return b''.join(chunks)
    def readline(self, size=-1):
        self._checkClosed();chunks=[];remaining=size
        while remaining!=0:
            stream=self._stream()
            if stream is None:break
            data=stream.readline(remaining);self.position+=len(data);chunks.append(data)
            if remaining>0:remaining-=len(data)
            if data.endswith(b'\n'):break
        return b''.join(chunks)
    def close(self):
        if self.stream:self.stream.close()
        super().close()


class SplitFiles:
    def __init__(self, paths):self.paths=paths
    def open(self, mode='rb'):
        if mode!='rb':raise ValueError('Split shards are read-only binary streams')
        return JoinedReader(self.paths)
    def stat(self):return SimpleNamespace(st_size=sum(p.stat().st_size for p in self.paths))


def split_file(package, split, summary=None):
    package=Path(package).resolve()
    if summary is None:summary=json.loads((package/'dataset_metadata.json').read_text())
    refs=summary.get('split_files',{}).get(split,split+'.jsonl')
    if isinstance(refs,str):refs=[refs]
    if not isinstance(refs,list) or not refs:raise ValueError('Missing split shards')
    paths=[resource_path(package,{'file':ref},'file') for ref in refs]
    if len(set(paths))!=len(paths):raise ValueError('Duplicate split shards')
    return paths[0] if len(paths)==1 else SplitFiles(paths)


def verify_package_provenance(package, summary):
    from .predictor_judge_dataset import file_hash
    package=Path(package)
    for field,name in [('inventory_sha256','episode_inventory.jsonl'),
                       ('source_provenance_sha256','source_provenance.json')]:
        if field in summary and file_hash(package/name)!=summary[field]:
            raise ValueError('Package '+name+' bytes changed')
    for relative,digest in summary.get('shard_metadata_sha256',{}).items():
        path=resource_path(package,{'file':relative},'file')
        if file_hash(path/'dataset_metadata.json')!=digest:raise ValueError('Shard metadata changed')
        child=json.loads((path/'dataset_metadata.json').read_text())
        if child.get('shard_metadata_sha256'):raise ValueError('Nested shard catalogs are unsupported')
        verify_package_provenance(path,child)


def _share_file(source, target):
    from .predictor_judge_dataset import file_hash
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_hash(source)!=file_hash(target):raise ValueError('Shared resource content differs')
        return
    try:os.link(source,target)
    except OSError as exc:
        import errno
        if exc.errno!=errno.EXDEV:raise
        shutil.copyfile(source,target)


def publish_node_package(package, root, task, node, source_paths):
    """Publish derived files only; raw locations are an explicit delivery mapping."""
    package=Path(package).resolve();root=Path(root).resolve()
    if task not in ('classifier','predictor_judge') or not node or Path(node).name!=node:
        raise ValueError('Invalid package namespace')
    if (package/'BUILDING').exists():raise ValueError('Unfinished package')
    meta=json.loads((package/'dataset_metadata.json').read_text())
    if meta.get('task')!=task:raise ValueError('Package task differs')
    verify_package_provenance(package,meta)
    output=root/task/'shards'/node;output.mkdir(parents=True,exist_ok=False)
    (output/'BUILDING').touch()
    for eid,res in meta['resources'].items():
        for field,folder in [('record_file','records'),('media_manifest','media'),('annotation_file','annotations')]:
            source=resource_path(package,res,field)
            target=root/'shared'/folder/node/source.name
            _share_file(source,target);res[field]=os.path.relpath(target,output)
        source_ref=Path(source_paths[eid])
        if source_ref.is_absolute() or '..' in source_ref.parts:raise ValueError('Invalid source delivery path')
        res['raw_root']=os.path.relpath(root/source_ref,output);res['dataset_root']='../../..'
    for name in ['train.jsonl','validation.jsonl','test.jsonl','excluded.jsonl',
                 'semantic_definitions.json','composition.json','DATA_SUMMARY.md','source_provenance.json']:
        if (package/name).exists():_share_file(package/name,output/name)
    inventory=[json.loads(line) for line in (package/'episode_inventory.jsonl').read_text().splitlines()]
    for row in inventory:
        relative=Path(source_paths[row['episode_id']])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid inventory source path')
        row['raw_root']=os.path.relpath(root/relative,output)
    (output/'episode_inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inventory))
    from .predictor_judge_dataset import file_hash
    meta['inventory_sha256']=file_hash(output/'episode_inventory.jsonl')
    (output/'dataset_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'BUILDING').unlink()
    return meta


def combine_package_shards(package, nodes):
    """Make one logical package without concatenating or duplicating JSONL bytes."""
    from .predictor_judge_dataset import file_hash
    from .source_manifest import combine_training_normalization
    package=Path(package).resolve()
    if not nodes or len(set(nodes))!=len(nodes):raise ValueError('Unique node shards required')
    shards=[]
    for node in nodes:
        path=resource_path(package,{'file':'shards/'+node},'file')
        if (path/'BUILDING').exists():raise ValueError('Unfinished shard')
        shards.append((path,json.loads((path/'dataset_metadata.json').read_text())))
    if len({('source_provenance_sha256' in s) for _,s in shards})!=1:
        raise ValueError('All shards must provide the same provenance contract')
    meta=copy.deepcopy(shards[0][1]);resources={};counts=Counter();reasons=Counter()
    inventory=[];inventory_ids=set();reports=[];cohort_identity=None
    invariant=('method','task','status','input_contract','history_frames','max_actions','state_features',
               'semantic_definitions_sha256','group_splits','selection')
    for path,shard in shards:
        verify_package_provenance(path,shard)
        if any(shard.get(k)!=meta.get(k) for k in invariant):raise ValueError('Shard protocols or splits differ')
        records=[json.loads(line) for line in (path/'episode_inventory.jsonl').read_text().splitlines()]
        for row in records:
            if row['episode_id'] in inventory_ids:raise ValueError('Duplicate inventory episode across shards')
            inventory_ids.add(row['episode_id'])
            row['raw_root']=os.path.relpath((path/row['raw_root']).resolve(),package)
            inventory.append(row)
        if 'source_provenance_sha256' in shard:
            origin=json.loads((path/'source_provenance.json').read_text())
            identity=(origin['source_manifest_sha256'],origin['source_manifest_episodes'])
            if cohort_identity is not None and identity!=cohort_identity:raise ValueError('Source cohorts differ')
            cohort_identity=identity
            if {r['episode_id'] for r in origin['selected']}!={r['episode_id'] for r in records}:
                raise ValueError('Source provenance and inventory differ')
            if origin.get('build_group_splits',meta['group_splits'])!=meta['group_splits']:
                raise ValueError('Source provenance build splits differ')
        reports.append(json.loads((path/'composition.json').read_text()))
        if file_hash(path/'semantic_definitions.json')!=shard['semantic_definitions_sha256']:
            raise ValueError('Shard semantic definitions changed')
        for split,digest in shard['file_sha256'].items():
            if file_hash(split_file(path,split,shard))!=digest:raise ValueError('Shard split bytes changed')
        for eid,res in shard['resources'].items():
            if eid in resources:raise ValueError('Duplicate episode across shards')
            item=copy.deepcopy(res)
            for field in ('record_file','media_manifest','annotation_file','raw_root'):
                item[field]=os.path.relpath(resource_path(path,res,field),package)
            item['dataset_root']='..';resources[eid]=item
        counts.update(shard['counts']);reasons.update(shard['label_reasons'])
    meta.update(resources=resources,counts=dict(counts),label_reasons=dict(reasons),
                normalization=combine_training_normalization([s for _,s in shards]))
    meta['normalization_statistics']={}
    for key in ('state','action'):
        stats=[s['normalization_statistics'][key] for _,s in shards]
        meta['normalization_statistics'][key]=dict(count=sum(s['count'] for s in stats),
            sum=[sum(v) for v in zip(*(s['sum'] for s in stats))],
            sum_squares=[sum(v) for v in zip(*(s['sum_squares'] for s in stats))])
    meta.pop('source_provenance_sha256',None);meta.pop('inventory_sha256',None)
    meta['shard_metadata_sha256']={str(p.relative_to(package)):file_hash(p/'dataset_metadata.json') for p,_ in shards}
    meta['split_files']={split:[str((path/(split+'.jsonl')).relative_to(package)) for path,_ in shards]
                         for split in ('train','validation','test','excluded')}
    meta['file_sha256']={s:file_hash(split_file(package,s,meta)) for s in meta['split_files']}
    from .predictor_judge_curation import composition_report,write_summary
    from .semantic_reporting import merge_window_reports
    windows={}
    for report in reports:
        for field,groups in report['windows'].items():
            for group,values in groups.items():windows.setdefault(field,{}).setdefault(group,Counter()).update(values)
    report=composition_report(inventory,windows)
    report.update(supervision='semantic_safety',selection=meta['selection'])
    report['events']['definition']=reports[0]['events']['definition']
    for field,keys in [('semantic_windows',('split','family','semantic_id')),('query_windows',('split','family','query_id'))]:
        report[field]=merge_window_reports(reports,field,keys)
    report['state_segments']=[r for sub in reports for r in sub.get('state_segments',[])]
    report['state_segment_scope']=reports[0].get('state_segment_scope','not recorded by this builder')
    if sum(r['eligible'] for r in report['query_windows'])!=sum(counts.get(s,0) for s in ('train','validation','test')):
        raise ValueError('Composition eligible counts differ')
    if sum(r['excluded'] for r in report['query_windows'])!=counts.get('excluded',0):
        raise ValueError('Composition excluded counts differ')
    (package/'BUILDING').touch()
    (package/'episode_inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inventory))
    meta['inventory_sha256']=file_hash(package/'episode_inventory.jsonl')
    if cohort_identity is not None:meta['source_cohort']=dict(sha256=cohort_identity[0],episodes=cohort_identity[1])
    (package/'composition.json').write_text(json.dumps(report,indent=2)+'\n')
    write_summary(package/'DATA_SUMMARY.md',report)
    _share_file(shards[0][0]/'semantic_definitions.json',package/'semantic_definitions.json')
    temporary=package/'dataset_metadata.tmp'
    temporary.write_text(json.dumps(meta,indent=2)+'\n');temporary.replace(package/'dataset_metadata.json')
    (package/'BUILDING').unlink()
    return meta


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True)
    parser.add_argument('--nodes',nargs='+',required=True)
    args=parser.parse_args(argv)
    result=combine_package_shards(args.package,args.nodes)
    print(json.dumps({'counts':result['counts'],'episodes':len(result['resources'])}))


if __name__=='__main__':main()
