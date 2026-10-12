"""Opt-in acceleration around the unchanged native multimodal checkpoint loader.

FLA follows the first optimization in open-jev-fast (c52b8bb); its text-only
FastQwen35 backend is not used. No LoRA merge, quantization or mask changes.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json

UPSTREAM = {'repo': 'https://github.com/lyuyiqi/open-jev-fast',
            'revision': 'c52b8bb958c1f0d241d4eb7fce4ecd8d885bf1e4',
            'scope': 'FLA chunk recurrence; not the upstream text-only CUDA backend'}


@contextmanager
def reference_construction():
    """Installed optional packages must not silently change reference numerics."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ('FusedRMSNormGated', 'causal_conv1d_fn', 'causal_conv1d_update',
             'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule')
    previous = {name: getattr(qwen, name) for name in names}
    try:
        for name in names:
            setattr(qwen, name, None)
        yield
    finally:
        for name, value in previous.items():
            setattr(qwen, name, value)


def configure_attention(model, backend):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if backend not in ('torch', 'fla'):
        raise ValueError('Unsupported attention backend')
    function = qwen.torch_chunk_gated_delta_rule
    version = None
    if backend == 'fla':
        version = importlib.metadata.version('flash-linear-attention')
        if version != '0.5.2':
            raise ValueError('FLA backend requires flash-linear-attention==0.5.2')
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        function = chunk_gated_delta_rule
    layers = [m for m in model.backbone.language_model.modules()
              if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
    if not layers:
        raise ValueError('Expected native Qwen Gated DeltaNet layers')
    for layer in layers:
        layer.chunk_gated_delta_rule = function
    return {'attention_backend': backend, 'fla_version': version, 'linear_layers': len(layers)}


def feature_key(args, kwargs):
    """Hash actual processed tensors, including grid and options, not raw images."""
    import torch
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape), str(tensor.device)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif value is None or type(value) in (bool, int, float, str):
            digest.update(json.dumps([type(value).__name__, value], allow_nan=False).encode())
        elif isinstance(value, (tuple, list)):
            digest.update(json.dumps([type(value).__name__, len(value)]).encode())
            for item in value: add(item)
        elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
            digest.update(json.dumps(['dict', sorted(value)]).encode())
            for key in sorted(value): add(value[key])
        else:
            raise TypeError('Unsupported feature-cache argument')
    add(args); add(kwargs)
    return digest.hexdigest()


class VisionFeatureCache:
    """One-entry inference cache, cleared at every predict request and on errors."""
    def __init__(self, backbone):
        self.backbone = backbone
        self.original = backbone.get_image_features
        self.enabled = False
        self.clear()
        backbone.get_image_features = self.call

    def clear(self):
        self.key = self.value = None
        self.hits = self.misses = 0

    def call(self, *args, **kwargs):
        import torch
        if not self.enabled or self.backbone.training or torch.is_grad_enabled():
            self.clear()
            return self.original(*args, **kwargs)
        try:
            key = feature_key(args, kwargs)
        except (TypeError, ValueError):
            self.clear()
            return self.original(*args, **kwargs)
        if key == self.key:
            self.hits += 1
            return copy.copy(self.value)
        self.key = self.value = None
        self.misses += 1
        result = self.original(*args, **kwargs)
        self.key, self.value = key, result
        return copy.copy(result)
