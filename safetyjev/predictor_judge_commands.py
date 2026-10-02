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
    args=list(benchmark_args)
    if args[:1]==['--']:args=args[1:]
    if any(a.split('=')[0].startswith('--recording-') for a in args):raise ValueError('Recording options are set by the capture command')
    cmd=[sys.executable,'-m','maniguard.eval.benchmark',*args,'--recording-factory','safetyjev.predictor_judge_capture:create_observer',
         '--recording-output-dir',str(Path(output).resolve()),'--recording-provenance',str(Path(provenance).resolve())]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(root),str(Path(__file__).resolve().parents[1]),env.get('PYTHONPATH','')])
    return cmd,env


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--maniguard-root',required=True);cap.add_argument('--output',required=True);cap.add_argument('--provenance',required=True);cap.add_argument('benchmark_args',nargs=argparse.REMAINDER)
    build=sub.add_parser('build');build.add_argument('--episodes',required=True);build.add_argument('--output',required=True);build.add_argument('--history-frames',type=int,default=3);build.add_argument('--seed',type=int,default=42)
    build.add_argument('--split-manifest',help='Frozen JSON mapping base-task group to train/validation/test')
    build.add_argument('--unsafe-per-safe',type=float,default=4)
    build.add_argument('--active-motion-rad',type=float,default=.05)
    a=p.parse_args(argv)
    if a.command=='capture':
        cmd,env=capture_command(a.maniguard_root,a.output,a.provenance,a.benchmark_args)
        return subprocess.call(cmd,cwd=a.maniguard_root,env=env)
    directories=sorted({f.parent for name in ('record.json','episode.json','episode_status.json') for f in Path(a.episodes).glob('*/'+name)})
    if not directories:raise ValueError('No captured episode records')
    assignment=json.loads(Path(a.split_manifest).read_text()) if a.split_manifest else None
    build_package(directories,a.output,history_frames=a.history_frames,seed=a.seed,group_splits=assignment,
                  unsafe_per_safe=a.unsafe_per_safe,active_motion_rad=a.active_motion_rad);return 0

if __name__=='__main__':raise SystemExit(main())
