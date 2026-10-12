"""Configuration for the native Unsafe600 loop; secrets stay process-local."""
import copy
import json
import math
from pathlib import Path
from urllib.parse import urlparse

REVISION = '7c1c0fc46227eca60473577a4450b86f93403603'
MODEL_PATH = 'predictor-judge-27b-unsafe600/step-00000500/model'
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
IDENTITY = {'checkpoint_repo': 'IDEAS-Lab-Northwestern/SafetyJev-Checkpoints',
            'checkpoint_revision': REVISION, 'checkpoint_subdirectory': MODEL_PATH,
            'base_model_revision': BASE_REVISION, 'input_contract': 'chunk_start_v2',
            'loader_sha256': '013c4c1c62a657472ddf680d7db7993df511351a333b8d6e6fc66f7b9707ce1f'}
DEFAULTS = {
    'local_retries': 2, 'repairs_per_decision': 1, 'execution_horizon': 8,
    'score': 'satisfied_constraints', 'acceptance': 'all_no',
    'fallback': 'best_valid', 'tie_break': 'earliest', 'constraint_ids': None,
    'predictor': {'endpoint': 'http://127.0.0.1:8794', 'timeout': 120,
                  'checkpoint_dir': '../checkpoints/unsafe600', 'hf_token': '', 'batch_size': 1,
                  'attention_backend': 'torch', 'vision_cache': False},
    'system1': {'enabled': True, 'model': 'deepseek/deepseek-v4.1-flash', 'api_key': '',
                'include_images': True, 'reasoning_enabled': False, 'timeout': 30,
                'max_tokens': 256, 'max_calls_per_episode': 3, 'history_limit': 8},
}


def validate_config(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown agentic config fields')
    result = copy.deepcopy(DEFAULTS)
    for key, item in value.items():
        if key in ('predictor', 'system1'):
            if not isinstance(item, dict) or set(item)-set(DEFAULTS[key]):
                raise ValueError('Unknown fields in ' + key)
            result[key].update(item)
        else:
            result[key] = item
    for key, maximum in [('local_retries', 8), ('repairs_per_decision', 4)]:
        if type(result[key]) is not int or not 0 <= result[key] <= maximum:
            raise ValueError('Invalid ' + key)
    for key in ('execution_horizon', 'score', 'acceptance', 'fallback', 'tie_break'):
        if type(result[key]) is not type(DEFAULTS[key]) or result[key] != DEFAULTS[key]:
            raise ValueError('Unsupported ' + key + ' for this checkpoint')
    ids = result['constraint_ids']
    if ids is not None and (not isinstance(ids, list) or not ids or
            any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids)):
        raise ValueError('constraint_ids must be null or distinct trained query IDs')
    pred, planner = result['predictor'], result['system1']
    if not isinstance(pred['endpoint'], str):
        raise ValueError('Invalid predictor endpoint')
    url = urlparse(pred['endpoint'])
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
        raise ValueError('Predictor must use a local HTTP endpoint')
    if url.port is None or not 1 <= url.port <= 65535:
        raise ValueError('Predictor endpoint requires an explicit port')
    if not isinstance(pred['checkpoint_dir'], str) or not pred['checkpoint_dir'].strip():
        raise ValueError('Invalid checkpoint directory')
    if type(pred['batch_size']) is not int or not 1 <= pred['batch_size'] <= 8:
        raise ValueError('Invalid predictor batch_size')
    if pred['attention_backend'] not in ('torch', 'fla'):
        raise ValueError('Invalid predictor attention_backend')
    if type(pred['vision_cache']) is not bool:
        raise ValueError('Invalid predictor vision_cache')
    for section in (pred, planner):
        if type(section['timeout']) not in (int, float) or not math.isfinite(section['timeout']) or section['timeout'] <= 0:
            raise ValueError('Invalid service timeout')
    if type(planner['enabled']) is not bool:
        raise ValueError('system1.enabled must be Boolean')
    if type(planner['max_calls_per_episode']) is not int or not 0 <= planner['max_calls_per_episode'] <= 1000:
        raise ValueError('Invalid System 1 episode budget')
    for key, maximum in (('max_tokens', 4096), ('history_limit', 64)):
        if type(planner[key]) is not int or not 1 <= planner[key] <= maximum:
            raise ValueError('Invalid System 1 ' + key)
    for key in ('include_images', 'reasoning_enabled'):
        if type(planner[key]) is not bool:
            raise ValueError('Invalid System 1 ' + key)
    if not isinstance(planner['model'], str) or '/' not in planner['model'] or not planner['model'].strip():
        raise ValueError('Invalid OpenRouter model ID')
    for section, key in ((pred, 'hf_token'), (planner, 'api_key')):
        if not isinstance(section[key], str):
            raise ValueError('Credential fields must be strings')
    return result


def load_config(path):
    path = Path(path).resolve()
    config = validate_config(json.loads(path.read_text()))
    checkpoint = Path(config['predictor']['checkpoint_dir']).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    config['predictor']['checkpoint_dir'] = str(checkpoint.resolve())
    return config


def public_config(config):
    result = copy.deepcopy(config)
    del result['predictor']['hf_token']
    del result['system1']['api_key']
    return result
