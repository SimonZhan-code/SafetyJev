"""NVIDIA-only serving for the pinned Unsafe600 chunk_start_v2 checkpoint."""
import argparse
import base64
import hashlib
import inspect
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
import time
from urllib.parse import urlparse

from .agentic_config import BASE_REVISION, IDENTITY, MODEL_PATH, load_config

NUMERIC_SHAPES = {'robot_state': (16,), 'history_robot_states': (8,16),
                  'executed_actions': (8,8), 'remaining_actions': (8,8)}
MASKS = ('history_mask', 'history_action_mask', 'action_mask')


def decode_inputs(inputs):
    import numpy as np
    from PIL import Image
    expected = set(NUMERIC_SHAPES) | set(MASKS) | {'observations','action_dt_s','constraint_context'}
    if not isinstance(inputs, dict) or set(inputs) != expected:
        raise ValueError('Invalid native input fields')
    arrays = {}
    for key, shape in NUMERIC_SHAPES.items():
        a = np.asarray(inputs[key], dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('Invalid numeric input: ' + key)
        arrays[key] = a
    for key in MASKS:
        m = inputs[key]
        if not isinstance(m, list) or len(m) != 8 or any(type(v) is not bool for v in m):
            raise ValueError('Invalid native mask')
        arrays[key] = np.asarray(m, dtype=bool)
    hm, pm = arrays['history_mask'], arrays['history_action_mask']
    if (not arrays['action_mask'].all() or not hm[-1] or (pm & ~hm).any() or
            any(not np.array_equal(m, np.arange(8) >= 8-m.sum()) for m in (hm, pm)) or
            (hm.sum()-pm.sum()) > 1):
        raise ValueError('Invalid history/action alignment')
    if type(inputs['action_dt_s']) not in (int,float) or not np.isfinite(inputs['action_dt_s']) or inputs['action_dt_s'] <= 0:
        raise ValueError('Invalid control interval')
    if inputs['constraint_context'] != {'source':'none','text':''}:
        raise ValueError('External constraint context is not supported by this checkpoint')
    arrays['action_dt_s'] = np.asarray(inputs['action_dt_s'], dtype=np.float32)
    views = inputs['observations']
    if not isinstance(views, dict) or set(views) != {'overview','wrist'}:
        raise ValueError('Both native cameras are required')
    arrays['observations'] = {}
    for camera, frames in views.items():
        if not isinstance(frames, list) or len(frames) != 8:
            raise ValueError('Expected eight image slots')
        decoded = []
        for j, blob in enumerate(frames):
            if not hm[j]:
                if blob is not None:
                    raise ValueError('Masked image slots must be null')
                decoded.append(None)
                continue
            data = base64.b64decode(blob, validate=True)
            with Image.open(BytesIO(data)) as im:
                if im.width*im.height > 1024*1024:
                    raise ValueError('Image too large')
                decoded.append(np.array(im.convert('RGB'), copy=True).transpose(2,0,1))
        template = decoded[-1]
        if any(frame is not None and frame.shape != template.shape for frame in decoded):
            raise ValueError('Inconsistent image shapes')
        arrays['observations'][camera] = np.stack([np.zeros_like(template) if f is None else f for f in decoded])
    return arrays


def checkpoint_files(config, download=False):
    root = Path(config['predictor']['checkpoint_dir'])
    token = config['predictor']['hf_token']
    if token:
        os.environ['HF_TOKEN'] = token
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(IDENTITY['checkpoint_repo'], revision=IDENTITY['checkpoint_revision'], token=token or None,
                          local_dir=root, allow_patterns=[MODEL_PATH+'/*', MODEL_PATH.rsplit('/',1)[0]+'/checkpoint.json'])
    checkpoint = root/MODEL_PATH
    manifest = json.loads((checkpoint.parent/'checkpoint.json').read_text())
    files = manifest['files_sha256']
    required = {'model/model.json', 'model/head.pt', 'model/predictor_judge.pt'}
    if not required.issubset(files) or not any(n.startswith('model/processor/') for n in files) or not any(n.startswith('model/adapter/') for n in files):
        raise ValueError('Incomplete native checkpoint manifest')
    for name, expected in files.items():
        if name.startswith('model/'):
            path = (checkpoint.parent/name).resolve()
            if checkpoint.resolve() not in path.parents:
                raise ValueError('Invalid checkpoint manifest path')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('Checkpoint file digest mismatch: ' + name)
    return checkpoint


class NativeService:
    def __init__(self, config, download=False):
        import torch
        import jev.predictor_judge_model as upstream
        import jev.visual_model as visual
        from . import chunk_judge_model
        if not torch.cuda.is_available() or torch.version.hip:
            raise ValueError('This integration requires an NVIDIA CUDA device')
        sources = [(chunk_judge_model, IDENTITY['loader_sha256']),
                   (upstream, 'b8ccb37389b3439b7704abe71141e4c2862fd6a31797b39c595ffcba2953d2a1'),
                   (visual, 'bb8fd3c659eae324b70ff872fafe0d58c03a1dbbec26dfe3eaaeb2d84e9844a0')]
        for module, digest in sources:
            if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != digest:
                raise ValueError('Native loader dependency hash mismatch')
        path = checkpoint_files(config, download)
        metadata = json.loads((path/'model.json').read_text())
        if (metadata['revision'] != BASE_REVISION or metadata['input_contract'] != 'chunk_start_v2'
                or metadata['state_dim'] != 16 or metadata['history_frames'] != 8 or metadata['max_actions'] != 8):
            raise ValueError('Native checkpoint contract mismatch')
        from .inference_optimizations import reference_construction, VisionFeatureCache
        with reference_construction():
            self.model = chunk_judge_model.ChunkJudgeModel.load(path, device='cuda:0')
        self.vision_cache = VisionFeatureCache(self.model.backbone)
        self.batch_size = config['predictor']['batch_size']
        self.health = {**IDENTITY, 'status': 'ready', 'device': torch.cuda.get_device_name(),
                       'training_loader_sha256': 'd33dca3d21c7cf6a73432536fc71e7759f7fab0a60557b7c40a446ec5b7ee39d',
                       'loader_note': 'Published NVIDIA loader; unpublished AMD patch not applied',
                       'checkpoint_manifest_sha256': hashlib.sha256((path.parent/'checkpoint.json').read_bytes()).hexdigest(),
                       'batch_size': self.batch_size}
        self.configure_optimizations(config['predictor'].get('attention_backend', 'torch'),
                                     config['predictor'].get('vision_cache', False))

    def configure_optimizations(self, attention_backend, vision_cache):
        from .inference_optimizations import configure_attention, UPSTREAM
        settings = configure_attention(self.model, attention_backend)
        self.vision_cache.clear()
        self.vision_cache.enabled = vision_cache
        self.health['optimizations'] = {**settings, 'vision_cache': vision_cache,
            'upstream': UPSTREAM, 'lora_merged': False, 'quantized': False}

    def predict(self, body):
        import torch
        if set(body) != {'request_id','questions','inputs'} or not isinstance(body['request_id'], str):
            raise ValueError('Invalid native request')
        questions = body['questions']
        if (not isinstance(questions, list) or not 1 <= len(questions) <= 32 or
                any(not isinstance(q, dict) or set(q) != {'id','question'} or
                    not isinstance(q['id'], str) or not q['id'] or
                    not isinstance(q['question'], str) or not 1 <= len(q['question']) <= 4096 for q in questions) or
                len({q['id'] for q in questions}) != len(questions)):
            raise ValueError('Invalid question batch')
        arrays = decode_inputs(body['inputs'])
        self.vision_cache.clear()
        answers = {}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(questions), self.batch_size):
                batch = questions[offset:offset+self.batch_size]
                inputs = {k: torch.from_numpy(v).unsqueeze(0).expand(len(batch), *v.shape)
                          for k, v in arrays.items() if k != 'observations'}
                inputs['observations'] = {k: torch.from_numpy(v).unsqueeze(0).expand(len(batch), *v.shape)
                                          for k, v in arrays['observations'].items()}
                logits = self.model(questions=[q['question'] for q in batch], **inputs)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite native logits')
                delta = (logits[:,1]-logits[:,0]).float().cpu().tolist()
                for q, value in zip(batch, delta):
                    answers[q['id']] = {'violated': value >= 0, 'logit': value}
        torch.cuda.synchronize()
        return {**self.health, 'request_id': body['request_id'], 'answers': answers,
                'latency_s': time.perf_counter()-started,
                'vision_cache': {'hits': self.vision_cache.hits, 'misses': self.vision_cache.misses}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    service = NativeService(config, args.download)
    # Full-history warmup checks image/token dimensions before serving readiness.
    from PIL import Image
    buffer = BytesIO(); Image.new('RGB',(256,256)).save(buffer, format='PNG')
    image = base64.b64encode(buffer.getvalue()).decode()
    inputs = {k: __import__('numpy').zeros(shape).tolist() for k, shape in NUMERIC_SHAPES.items()}
    inputs.update({k: [True]*8 for k in MASKS})
    inputs.update(observations={c:[image]*8 for c in ('overview','wrist')}, action_dt_s=.05,
                  constraint_context={'source':'none','text':''})
    service.predict({'request_id':'warmup','questions':[{'id':'warmup','question':'Is the jar tilted?'}], 'inputs':inputs})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, value, status=200):
            blob=json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length', str(len(blob))); self.end_headers(); self.wfile.write(blob)
        def do_GET(self):
            self.send(service.health if self.path == '/health' else {'error':'not_found'}, 200 if self.path == '/health' else 404)
        def do_POST(self):
            if self.path != '/predict':
                self.send({'error':'not_found'},404); return
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0 < length <= 64*1024*1024: raise ValueError('Invalid body size')
                self.send(service.predict(json.loads(self.rfile.read(length))))
            except Exception as exc:
                self.send({'error':type(exc).__name__},500)
    address=urlparse(config['predictor']['endpoint'])
    print(json.dumps(service.health), flush=True)
    HTTPServer(('127.0.0.1', address.port or 80), Handler).serve_forever()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit('Native predictor failed (' + type(exc).__name__ + ')') from None
