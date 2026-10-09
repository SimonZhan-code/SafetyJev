"""Build a SafetyJev inference bundle, then optionally publish to a new private model repo."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile

from .experiment_tracking import public_config,is_remote_model_id


MODEL_FILES = {
    'model.json', 'head.pt', 'predictor_judge.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
    'processor/processor_config.json', 'processor/preprocessor_config.json',
    'processor/tokenizer.json', 'processor/tokenizer_config.json',
    'processor/special_tokens_map.json', 'processor/added_tokens.json',
    'processor/vocab.json', 'processor/merges.txt', 'processor/chat_template.jinja',
    'processor/video_preprocessor_config.json',
}
METRIC_KEYS = ('evaluated', 'nll', 'brier', 'accuracy', 'confusion', 'answers', 'by_query',
               'temperature', 'ece', 'model_time_seconds', 'time_scope',
               'samples', 'micro', 'by_constraint', 'by_family_constraint', 'by_valid_steps',
               'events', 'safe_source_episodes', 'threshold', 'model_batch_seconds', 'evaluation_seconds',
               'evaluation_time_scope', 'interpretation', 'headline')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(model):
    """Hash inference files only; generated PEFT README is not a model input."""
    model = Path(model)
    result = {}
    for name in sorted(MODEL_FILES):
        path = model/name
        if path.exists():
            if not path.is_file() or not path.resolve().is_relative_to(model.resolve()):
                raise ValueError('Model artifact escapes its directory: '+name)
            result[name] = sha256(path)
    if not {'model.json', 'head.pt'}.issubset(result):
        raise ValueError('Missing model configuration or shared head')
    return result


def model_task(config):
    if config.get('method') == 'native_qwen_visual_noul' and config.get('history_frames') == 1:
        return 'classifier'
    if config.get('method') in ('native_qwen_action_conditioned_noul','native_qwen_chunk_noul'):
        return 'predictor_judge'
    raise ValueError('Expected a current-camera classifier or Predictor Judge checkpoint')


def validate_test_report(test, model, identity):
    task = model_task(json.loads((Path(model)/'model.json').read_text()))
    if test.get('task') != task or test.get('split') != 'test':
        raise ValueError('Expected a standalone '+task+' test report')
    count = test.get('samples') if task == 'predictor_judge' else test.get('evaluated')
    if not test.get('expected_samples') or count != test['expected_samples']:
        raise ValueError('Test report does not cover the complete split')
    if test.get('checkpoint_sha256') != checkpoint_hashes(model):
        raise ValueError('Test report was produced by another checkpoint')
    if test.get('package_sha256') != identity['package_sha256'] or test.get('split_sha256') != identity['split_hashes']['test']:
        raise ValueError('Test report uses a different dataset or split')


def build_export(run, test_report, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    final = json.loads((run/'final/report.json').read_text())
    if final['training']['status'] != 'completed':
        raise ValueError('Training has not completed')
    if final['identity']['evaluation'].get('max_batches') is not None:
        raise ValueError('Cannot export a capped development evaluation as a completed experiment')
    model = run/'final/model'
    config = json.loads((model/'model.json').read_text())
    task = model_task(config)
    if not is_remote_model_id(config.get('model_id')) or not re.fullmatch(r'[a-fA-F0-9]{40}', str(config.get('revision'))):
        raise ValueError('Export requires a remote backbone pinned to a full commit revision')
    test = json.loads(Path(test_report).read_text())
    validate_test_report(test, model, final['identity'])
    hashes = checkpoint_hashes(model)
    required = {'processor/processor_config.json', 'processor/tokenizer.json'}
    if task == 'predictor_judge':
        required.add('predictor_judge.pt')
    if config.get('lora_rank'):
        required |= {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    if not required.issubset(hashes):
        raise ValueError('Missing processor, adapter or Predictor Judge numeric weights')
    # Unknown inference files must be reviewed, not silently dropped.
    extra = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()} - MODEL_FILES - {'adapter/README.md'}
    if extra:
        raise ValueError('Unrecognized model files: '+', '.join(sorted(extra)))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.export-', dir=output.parent))
    try:
        for name in hashes:
            destination = stage/'model'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model/name, destination)
            if name.endswith('.json') and name != 'processor/tokenizer.json':
                metadata = json.loads(destination.read_text())
                # These are loader/cache provenance fields, not tokenizer or
                # adapter parameters. Keep inference settings unchanged.
                for key in ('name_or_path', '_name_or_path', 'cache_dir'):
                    metadata.pop(key, None)
                if name == 'adapter/adapter_config.json':
                    metadata['base_model_name_or_path'] = config['model_id']
                destination.write_text(json.dumps(metadata, indent=2)+'\n')
        metrics = {split: {k: report[k] for k in METRIC_KEYS if k in report}
                   for split, report in [('validation', final['validation']), ('test', test)]}
        (stage/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
        run_config = json.loads((run/'run.json').read_text())['config']
        identity = final['identity']
        provenance = {'task': task, 'answer_order': ['No', 'Yes'],
                      'config': public_config(run_config), 'source_checkpoint_sha256': hashes,
                      'training': {k: final['training'][k] for k in ('completed_step', 'best_step', 'best_nll') if k in final['training']},
                      'data': {k: identity[k] for k in ('package_sha256', 'split_hashes')},
                      'sources': identity.get('sources', {}), 'runtime': identity.get('runtime', {})}
        (stage/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
        classifier_card = '''---
library_name: peft
tags:
- safetyjev
- visual-classifier
---
# SafetyJev visual classifier

Current overview and wrist RGB images plus a natural-language AP question yield
No/Yes probabilities from a shared decision head. Yes is question-dependent and
is not universally unsafe. This is not an action-conditioned Predictor Judge.

`model/` contains the language adapter, shared head and processor. The pinned
base model in `model/model.json` is downloaded separately. Load this directory
with `jev.visual_model.VisualDecisionModel.load` from the corresponding SafetyJev
Open-Jev fork. It is not a standalone AutoModel or text-generation checkpoint.

`metrics.json` contains full validation/test results; `provenance.json` records
training parameters and data/source hashes. The five-family dataset consists
of reviewed unsafe episodes and their normal frames, so test results do not
measure false alarms in a natural population of wholly safe episodes. Some
questions have missing positive/negative support; inspect per-question metrics.

Preserve all applicable base-model and dataset terms when using this artifact.
'''
        judge_card = '''---
library_name: peft
tags:
- safetyjev
- predictor-judge
---
# SafetyJev Predictor Judge

Adjacent overview/wrist RGB observations, current robot state, masked remaining
commands and a natural-language constraint yield shared No/Yes scores. Yes means
a new constraint violation during the valid remaining command suffix. No does
not certify the whole task. Monitor states and AP truth are supervision only.

`model/` contains the language adapter, shared head, processor and
`predictor_judge.pt` (state/action projections and training normalization).
The pinned backbone in `model/model.json` is downloaded separately. Load with
`jev.predictor_judge_model.PredictorJudgeModel.load` from the corresponding
SafetyJev Open-Jev fork. This is not a standalone AutoModel checkpoint.

`metrics.json` contains full validation/test window, constraint and event metrics;
`provenance.json` records training parameters and data/source hashes. Training
samples are event-balanced; validation/test retain the supplied distribution.
Adjacent windows are correlated, and constraints without positive examples have
no recall evidence. Event metrics cover only events with eligible windows.
Scores are not established deployment probabilities or evidence of intervention
benefits. Read the recorded threshold when interpreting classification metrics.

Preserve all applicable base-model and dataset terms when using this artifact.
'''
        if config.get('input_contract')=='chunk_start_v2':
            judge_card='''# SafetyJev chunk-boundary Predictor Judge

The previous executed action segment, its post-action overview/wrist images and
aligned historical robot states, plus current proprio, eight future committed commands and one safety query produce
shared No/Yes scores. Yes means a semantic violation during the future segment,
including an ongoing state violation. Check once before each new segment.

Load with `safetyjev.chunk_judge_model.load_judge`. This is not a standalone
AutoModel checkpoint. `model/model.json` pins the separately obtained backbone;
`predictor_judge.pt` stores numeric projections and train-only normalization.
The chunk_start_v2 contract is incompatible with v1 chunk and legacy per-step
checkpoints. Historical and current states share train-only normalization and
the state projection. Metrics retain actual model-call and evaluation durations.

`metrics.json` retains compact headline confusion metrics and detailed diagnostics;
`provenance.json` records source/data identity. Training sampling uses episode/status
units, not independent physical event counts. Validation/test retain their supplied
distribution. Shadow replay is not evidence of intervention benefit, and an
unsafe-only cohort cannot establish safe-episode false-alarm rates.

Preserve all applicable base-model and dataset terms when using this artifact.
'''
        (stage/'README.md').write_text(judge_card if task == 'predictor_judge' else classifier_card)
        files = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'manifest.json').write_text(json.dumps({'files_sha256': files}, indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {'files': len(files)+1, 'output': str(output)}


def upload_export(folder, repo_id, *, api=None):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())['files_sha256']
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'manifest.json'}:
        raise ValueError('Export contains missing or unexpected files')
    for name, expected in manifest.items():
        path = folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or sha256(path) != expected:
            raise ValueError('Export checksum mismatch: '+name)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    # A new destination is deliberate: never overwrite another experiment or
    # accidentally publish into an existing public repository.
    api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=False)
    return api.upload_folder(repo_id=repo_id, repo_type='model', folder_path=str(folder),
                             commit_message='Add model and evaluation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    for name in ('run', 'test-report', 'output'):
        build.add_argument('--'+name, required=True)
    upload = sub.add_parser('upload')
    upload.add_argument('--folder', required=True)
    upload.add_argument('--repo-id', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        print(json.dumps(build_export(args.run, args.test_report, args.output), indent=2))
    else:
        print(upload_export(args.folder, args.repo_id))


if __name__ == '__main__':
    main()
