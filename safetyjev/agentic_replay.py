"""NVIDIA replay of published samples; compare outputs without changing the loop."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from .agentic_config import IDENTITY, MODEL_PATH, load_config
from .agentic_server import NativeService
from .chunk_judge import chunk_numeric_inputs
from .io import read_jsonl, write_json


def sample_inputs(root, record, row):
    import h5py
    root = Path(root)
    manifest = json.loads((root/'image_manifest.json').read_text())
    inputs = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in chunk_numeric_inputs(record, row).items()}
    inputs['constraint_context'] = row['constraint_context']
    inputs['observations'] = {c: [] for c in ('overview', 'wrist')}
    with h5py.File(root/'trajectory.hdf5', 'r') as media:
        for step in row['history_steps']:
            for camera, frames in inputs['observations'].items():
                if step is None:
                    frames.append(None)
                    continue
                ref = record['observations'][step]['images'][camera]
                kind, name, channel, index = ref.split(':')
                if kind != 'hdf5' or name != 'trajectory.hdf5' or '/' in channel or int(index) < 0:
                    raise ValueError('Unsupported replay image reference')
                group = media['images/'+channel]
                start, end = group['offsets'][int(index):int(index)+2]
                blob = group['bytes'][start:end].tobytes()
                if hashlib.sha256(blob).hexdigest() != manifest[ref]:
                    raise ValueError('Replay image digest mismatch')
                frames.append(base64.b64encode(blob).decode())
    return inputs


def choose_samples(rows, all_samples=False):
    if all_samples:
        return rows
    # Exercise reset and full-history inputs for every released family.
    selected = {}
    for row in rows:
        phase = 'reset' if row['start_step'] == 0 else 'history'
        if phase == 'history' and not all(s is not None for s in row['history_action_steps']):
            continue
        selected.setdefault((row['family'], phase), row)
    return list(selected.values())


def comparison(row, saved, answer, latency):
    logit = answer['logit']
    if not math.isfinite(logit):
        raise ValueError('Nonfinite replay output')
    score = 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))
    reference = saved['score']
    if type(reference) not in (int,float) or not math.isfinite(reference) or not 0 <= reference <= 1:
        raise ValueError('Invalid saved score')
    return {'id':row['id'], 'family':row['family'], 'start_step':row['start_step'],
            'query_id':row['constraint_id'], 'reference_probability':reference, 'probability':score,
            'probability_abs_difference':abs(score-reference), 'logit':logit,
            'logit_difference':logit-math.log(reference/(1-reference)) if 0 < reference < 1 else None,
            'reference_violated':reference >= .5, 'violated':answer['violated'],
            'binary_disagreement':answer['violated'] != (reference >= .5), 'latency_s':latency}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--sample-root', help='Release test-sample directory; defaults to the checkpoint download tree')
    p.add_argument('--output', required=True)
    p.add_argument('--download', action='store_true', help='Download pinned model and sample files')
    p.add_argument('--all', action='store_true', help='Replay all released windows instead of reset/history smoke samples')
    p.add_argument('--probability-atol', type=float, default=.02)
    a = p.parse_args()
    if not math.isfinite(a.probability_atol) or not 0 <= a.probability_atol <= 1:
        p.error('Invalid numerical comparison tolerance')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    config = load_config(a.config)
    release = MODEL_PATH.split('/')[0]
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot_download(IDENTITY['checkpoint_repo'], revision=IDENTITY['checkpoint_revision'],
            token=config['predictor']['hf_token'] or None, local_dir=config['predictor']['checkpoint_dir'],
            allow_patterns=[release+'/test-sample/*'])
    root = Path(a.sample_root or Path(config['predictor']['checkpoint_dir'])/release/'test-sample')
    rows = choose_samples(read_jsonl(root/'windows.jsonl'), a.all)
    saved_rows = read_jsonl(root/'predictions-step500.jsonl')
    saved = {r['id']:r for r in saved_rows}
    if not rows or len(saved) != len(saved_rows) or any(r['id'] not in saved for r in rows):
        raise ValueError('Missing or duplicate replay references')
    episodes = {}
    for path in sorted(root.glob('episodes/*/*/record.json')):
        record = json.loads(path.read_text())
        if record['episode_id'] in episodes:
            raise ValueError('Duplicate replay episode')
        episodes[record['episode_id']] = (path.parent, record)
    service = NativeService(config, download=a.download)
    outputs = []
    for row in rows:
        directory, record = episodes[row['episode_id']]
        result = service.predict({'request_id':row['id'],
            'questions':[{'id':row['constraint_id'], 'question':row['question']}],
            'inputs':sample_inputs(directory, record, row)})
        outputs.append(comparison(row, saved[row['id']], result['answers'][row['constraint_id']], result['latency_s']))
    disagreements = sum(r['binary_disagreement'] for r in outputs)
    above = sum(r['probability_abs_difference'] > a.probability_atol for r in outputs)
    report = {'provenance':service.health, 'samples':len(outputs), 'all_samples':a.all,
              'probability_atol':a.probability_atol, 'above_tolerance':above,
              'binary_disagreements':disagreements,
              'max_probability_abs_difference':max(r['probability_abs_difference'] for r in outputs),
              'mean_probability_abs_difference':sum(r['probability_abs_difference'] for r in outputs)/len(outputs),
              'comparison_passed':not disagreements and not above,
              'reference_file_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in ('windows.jsonl','predictions-step500.jsonl')},
              'scope':'Numerical replay only. No live simulation, benchmark accuracy, or safety-improvement claim.',
              'results':outputs}
    write_json(a.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','provenance')}))
    if not report['comparison_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit('Native replay failed (' + type(exc).__name__ + ')') from None
