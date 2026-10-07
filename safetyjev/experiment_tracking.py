"""Optional rank-zero scalar telemetry; local training files remain authoritative."""
import importlib
import json
import math
import os
import re
from pathlib import Path
import uuid
import warnings


CONFIG_KEYS = {
    'model': ('model_id', 'revision', 'dtype', 'lora_rank', 'max_length', 'min_pixels', 'max_pixels', 'gradient_checkpointing'),
    'data': ('batch_size', 'num_workers', 'balance', 'prepare_in_workers', 'prefetch_factor'),
    'training': ('max_steps', 'lr', 'head_lr', 'weight_decay', 'warmup_steps', 'clip_grad_norm', 'brier_weight', 'eval_every', 'save_every', 'global_batch_size', 'accumulation'),
    'evaluation': ('max_batches', 'validation_samples', 'validation_negative_samples', 'run_test'),
}


def is_remote_model_id(value):
    return (isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?', value))
            and not Path(value).exists())


def public_config(config):
    """Only experiment parameters, never dataset paths or arbitrary config keys."""
    result = {key: config[key] for key in ('seed', 'deterministic') if key in config}
    for section, keys in CONFIG_KEYS.items():
        result[section] = {key: config[section][key] for key in keys if key in config.get(section, {})}
    if config.get('task') in ('classifier', 'predictor_judge'):
        result['task'] = config['task']
    sampling = config.get('data', {}).get('sampling')
    if isinstance(sampling, dict):
        result['data']['sampling'] = {k: sampling[k] for k in ('positive_fraction', 'samples_per_epoch') if k in sampling}
    if "model_id" in result["model"] and not is_remote_model_id(result["model"]["model_id"]):
        result["model"]["model_id"] = "local-model"
    return result


def numeric_metrics(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, dict):
            result.update(numeric_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[name] = item
    return result


class ExperimentTracker:
    def __init__(self, output, *, resume=False):
        self.output = Path(output)
        self.resume = bool(resume)
        self.run = None
        self.failed = False
        self.attempt = 0
        self.enabled = os.environ.get('SAFETYJEV_TRACKING', 'none') == 'wandb' and int(os.environ.get('RANK', '0')) == 0
        if os.environ.get('SAFETYJEV_TRACKING', 'none') not in ('none', 'wandb'):
            raise ValueError('SAFETYJEV_TRACKING must be none or wandb')

    def start(self, config):
        if not self.enabled:
            return
        state_path = self.output/'tracking.json'
        old = json.loads(state_path.read_text()) if state_path.exists() else None
        project = os.environ.get('WANDB_PROJECT') or (old or {}).get('project')
        entity = os.environ.get('WANDB_ENTITY') or (old or {}).get('entity')
        if not project:
            raise ValueError('Set WANDB_PROJECT when enabling tracking')
        if old and not self.resume:
            raise ValueError('Tracking state exists; resume the original run or use a new directory')
        if old and (project, entity) != (old['project'], old['entity']):
            raise ValueError('Cannot change tracking destination when resuming')
        run_id = old['id'] if old else uuid.uuid4().hex[:12]
        if os.environ.get('WANDB_RUN_ID') not in (None, run_id):
            raise ValueError('Run ID is managed by tracking.json; unset WANDB_RUN_ID')
        mode = os.environ.get('WANDB_MODE', 'online')
        if mode not in ('online', 'offline'):
            raise ValueError('Use WANDB_MODE=online/offline or SAFETYJEV_TRACKING=none')
        sdk = importlib.import_module('wandb')
        self.attempt = (old or {}).get('attempt', 0) + 1
        self.output.mkdir(parents=True, exist_ok=True)
        self.run = sdk.init(project=project, entity=entity, id=run_id, resume='allow', mode=mode,
                            name=os.environ.get('WANDB_NAME') or (old or {}).get('name') or self.output.name,
                            tags=[s.strip() for s in os.environ.get('WANDB_TAGS', '').split(',') if s.strip()],
                            dir=str(self.output), config=public_config(config),
                            settings=sdk.Settings(disable_git=True, disable_code=True, console='off'))
        self.run.define_metric('optimizer_step')
        self.run.define_metric('train/*', step_metric='optimizer_step')
        self.run.define_metric('validation/*', step_metric='optimizer_step')
        self.run.define_metric('final_validation/*', step_metric='optimizer_step')
        self.run.define_metric('test/*', step_metric='optimizer_step')
        state = {'id': run_id, 'project': project, 'entity': entity, 'attempt': self.attempt,
                 'name': os.environ.get('WANDB_NAME') or (old or {}).get('name') or self.output.name}
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(state_path)

    def _log(self, values):
        if self.run is None or self.failed:
            return
        try:
            # Do not use SDK step=optimizer_step: after a crash, uncheckpointed
            # optimizer steps may repeat. The attempt makes that replay explicit.
            self.run.log({**values, 'attempt': self.attempt})
        except Exception:
            self.failed = True
            warnings.warn('Experiment tracking failed; training continues with local logs. Check the SDK logs.', RuntimeWarning)

    def step(self, record):
        self._log({'optimizer_step': record['step'], **numeric_metrics(record, 'train')})

    def evaluation(self, metrics, step, split='validation'):
        self._log({'optimizer_step': step, **numeric_metrics(metrics, split)})

    def finish(self, status):
        if self.run is None:
            return
        try:
            self.run.summary['training_status'] = status
            self.run.finish(exit_code=1 if status == 'failed' else 0)
        except Exception:
            warnings.warn('Experiment tracking could not finish; local logs remain available.', RuntimeWarning)
