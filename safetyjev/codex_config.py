"""Separate, credential-free configuration for the robot-tools-only experiment."""
import copy
import json
import math
from pathlib import Path

from .agentic_config import validate_config as validate_native

DEFAULTS = {
    'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
    'proposal_limit': 3, 'decision_timeout_s': 120, 'execution_horizon': 8,
    'max_steps': 256, 'tool_sequence': 'individual', 'decision_target_s': None,
    'skill_max_chunks': 8, 'skill_replans_without_progress': 1, 'skill_call_limit': 64,
    'skill_joint_delta_limit': 0.2,
    'runtime_python': '/workspace/codex-harness-env/bin/python',
    'runtime_root': '/opt/safetyjev-codex',
    'runtime_user': 'safetyjev-codex',
    'runtime_home': '/home/safetyjev-codex',
    'socket_root': '/run/safetyjev-codex',
    'predictor': {'endpoint': 'http://127.0.0.1:8794', 'timeout': 120,
                  'attention_backend': 'fla', 'vision_cache': False, 'batch_size': 1},
}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Unknown Codex harness config fields')
    result = copy.deepcopy(DEFAULTS)
    for k, v in value.items():
        if k == 'predictor':
            if not isinstance(v, dict) or set(v) - set(DEFAULTS[k]):
                raise ValueError('Unknown harness predictor fields')
            result[k].update(v)
        else:
            result[k] = v
    if result['model'] != 'gpt-6-astra' or result['reasoning_effort'] not in ('low', 'medium', 'high'):
        raise ValueError('Unsupported Codex model or reasoning effort')
    if result['tool_sequence'] not in ('individual', 'paired', 'prepared', 'skills'):
        raise ValueError('Unknown tool sequence')
    if type(result['proposal_limit']) is not int or not 1 <= result['proposal_limit'] <= 10:
        raise ValueError('Invalid proposal limit')
    if type(result['execution_horizon']) is not int or result['execution_horizon'] != 8:
        raise ValueError('Predictor requires eight actions')
    if type(result['max_steps']) is not int or result['max_steps'] <= 0 or result['max_steps'] % 8:
        raise ValueError('Action cap must be positive and divisible by eight')
    t = result['decision_timeout_s']
    if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
        raise ValueError('Invalid decision deadline')
    target = result['decision_target_s']
    if target is not None and (type(target) not in (int, float) or not math.isfinite(target) or not 0 < target <= t):
        raise ValueError('Invalid soft decision target')
    if result['tool_sequence'] == 'prepared' and result['proposal_limit'] != 2:
        raise ValueError('Prepared mode requires one initial proposal plus one retry')
    for k, low, high in (('skill_max_chunks', 1, 8), ('skill_replans_without_progress', 0, 2), ('skill_call_limit', 1, 256)):
        if type(result[k]) is not int or not low <= result[k] <= high:
            raise ValueError('Invalid bounded skill budget')
    delta = result['skill_joint_delta_limit']
    if type(delta) not in (int, float) or not math.isfinite(delta) or not 0 < delta <= 0.2:
        raise ValueError('Invalid joint primitive bound')
    if result['tool_sequence'] == 'skills' and (result['proposal_limit'] != 2 or t > 120):
        raise ValueError('Skills require two proposals and watchdog at most 120 seconds')
    for k in ('runtime_python', 'runtime_root', 'runtime_home', 'socket_root'):
        if not isinstance(result[k], str) or not Path(result[k]).is_absolute():
            raise ValueError('Runtime paths must be absolute')
    if result['runtime_user'] != 'safetyjev-codex':
        raise ValueError('Use the dedicated unprivileged runtime account')
    p = result['predictor']
    validate_native({'predictor': p})
    if p['attention_backend'] != 'fla' or p['vision_cache'] or p['batch_size'] != 1:
        raise ValueError('This experiment pins FLA, batch one, cache off')
    return result


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


from .agentic_predictor import NativePredictor


class HarnessPredictor(NativePredictor):
    def validate_health(self, health):
        super().validate_health(health)
        if health.get('batch_size') != 1:
            raise ValueError('Codex experiment requires predictor batch size one')
