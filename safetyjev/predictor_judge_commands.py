"""Explicit predictor judge capture/build entrypoints; the legacy agentic adapter is unchanged."""
import argparse,ast,json,os,subprocess,sys
from pathlib import Path
from .predictor_judge_dataset import build_package


def capture_command(repo,output,provenance,benchmark_args):
    root=Path(repo).resolve();api=root/'maniguard/eval/recording.py'
    if not api.is_file():raise ValueError('This ManiGuard checkout has no passive recording hooks')
    tree=ast.parse(api.read_text());version=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RECORDING_API_VERSION' for t in node.targets):version=ast.literal_eval(node.value)
    if version!=1:raise ValueError('Unsupported ManiGuard recording API')
    if not (root/'maniguard/data/recording/observer.py').is_file():
        raise ValueError('This ManiGuard checkout lacks generic source recording')
    args=list(benchmark_args)
    if args[:1]==['--']:args=args[1:]
    if any(a.split('=')[0].startswith('--recording-') for a in args):raise ValueError('Recording options are set by the capture command')
    if not any(a.split('=')[0]=='--source-profile' for a in args):
        args += ['--source-profile',str(root/'configs/render/high_fidelity.yaml')]
    if not any(a.split('=')[0]=='--camera-resolution' for a in args):args += ['--camera-resolution','640']
    cmd=[sys.executable,'-m','maniguard.eval.benchmark',*args,'--recording-factory','maniguard.data.recording.observer:create_observer',
         '--recording-output-dir',str(Path(output).resolve()),'--recording-provenance',str(Path(provenance).resolve())]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(root),str(Path(__file__).resolve().parents[1]),env.get('PYTHONPATH','')])
    return cmd,env


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--maniguard-root',required=True);cap.add_argument('--output',required=True);cap.add_argument('--provenance',required=True);cap.add_argument('benchmark_args',nargs=argparse.REMAINDER)
    build=sub.add_parser('build');sources=build.add_mutually_exclusive_group(required=True)
    sources.add_argument('--episodes');sources.add_argument('--source-manifest',help='Frozen explicit source list; no directory scan')
    build.add_argument('--source-node',help='Node namespace from the frozen manifest')
    build.add_argument('--output',required=True);build.add_argument('--history-frames',type=int,default=3);build.add_argument('--seed',type=int,default=42)
    build.add_argument('--maniguard-root',help='Checkout containing the public source reader; only needed during source preparation')
    build.add_argument('--split-manifest',help='Frozen JSON mapping base-task group to train/validation/test')
    build.add_argument('--train-episodes',choices=['unsafe_only','unsafe_plus_safe','all'],default='unsafe_only',help='Training membership only; all retains the explicit source cohort regardless of derived labels')
    build.add_argument('--input-contract',choices=['per_step_v1','chunk_start_v2'],default='per_step_v1')
    build.add_argument('--unsafe-per-safe',type=float,default=4,help='Safe supplement ratio, used only with unsafe_plus_safe')
    build.add_argument('--active-motion-rad',type=float,default=.05)
    build.add_argument('--semantic-definitions',help='Explicit semantic supervision JSON; omission retains original monitor labels')
    build.add_argument('--semantic-task',choices=['classifier','predictor_judge'],default='predictor_judge')
    build.add_argument('--reuse-annotations',help='Verified annotations/ from an earlier semantic package over the same immutable sources')
    build.add_argument('--liquid-asset-root',help='Licensed source dataset root for offline liquid geometry; defaults to OMNIGIBSON_DATA_PATH')
    build.add_argument('--review',action='store_true',help='Build a marked development package from candidate semantic definitions')
    a=p.parse_args(argv)
    if a.command=='capture':
        cmd,env=capture_command(a.maniguard_root,a.output,a.provenance,a.benchmark_args)
        return subprocess.call(cmd,cwd=a.maniguard_root,env=env)
    if a.maniguard_root:sys.path.insert(0,str(Path(a.maniguard_root).resolve()))
    assignment=json.loads(Path(a.split_manifest).read_text()) if a.split_manifest else None
    provenance=None
    if a.source_manifest:
        from .source_manifest import read_source_manifest
        if not a.source_node or assignment is None:raise ValueError('Manifest requires source-node and explicit split')
        provenance=read_source_manifest(a.source_manifest,node=a.source_node,group_splits=assignment)
        directories=provenance.pop('directories')
    else:
        if a.source_node:raise ValueError('Source-node requires a source manifest')
        directories=sorted({f.parent for name in ('record.json','episode.json','episode_status.json') for f in Path(a.episodes).rglob(name)})
    if not directories:raise ValueError('No captured episode records')
    build_package(directories,a.output,history_frames=a.history_frames,seed=a.seed,group_splits=assignment,
                  unsafe_per_safe=a.unsafe_per_safe,active_motion_rad=a.active_motion_rad,train_episodes=a.train_episodes,
                  semantic_definitions=json.loads(Path(a.semantic_definitions).read_text()) if a.semantic_definitions else None,
                  semantic_task=a.semantic_task,review=a.review,reuse_annotations=a.reuse_annotations,liquid_asset_root=a.liquid_asset_root,
                  input_contract=a.input_contract,source_provenance=provenance);return 0

if __name__=='__main__':raise SystemExit(main())
