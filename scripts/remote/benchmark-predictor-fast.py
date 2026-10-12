"""Paired GPU replay of native and opt-in accelerated SafetyJev inference.

Input is a gzip JSON file containing exact saved native requests and metadata.
This loads one model; the GPU must have no other compute processes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetyjev.agentic_config import load_config
from safetyjev.agentic_predictor import request_digest

MODES = {'torch': ('torch', False), 'vision_cache': ('torch', True),
         'fla': ('fla', False), 'fla_vision_cache': ('fla', True)}


def summarize(samples, modes):
    result = {}
    for mode in modes:
        pairs = []
        for sample in samples:
            for repeat in sorted({m['repeat'] for m in sample['measurements']}):
                values = {m['mode']: m for m in sample['measurements'] if m['repeat'] == repeat}
                if mode in values and 'torch' in values:
                    pairs.append((values['torch'], values[mode]))
        if not pairs: continue
        flips = sum(a['answers'][q]['violated'] != b['answers'][q]['violated']
                    for a, b in pairs for q in a['answers'])
        acceptance = sum(all(not x['violated'] for x in a['answers'].values()) !=
                         all(not x['violated'] for x in b['answers'].values()) for a, b in pairs)
        delta = max(abs(a['answers'][q]['logit'] - b['answers'][q]['logit'])
                    for a, b in pairs for q in a['answers'])
        speedup = statistics.mean(a['wall_s'] for a, _ in pairs)/statistics.mean(b['wall_s'] for _, b in pairs)
        cache_verified = 'vision_cache' not in mode or all(
            b.get('vision_cache', {}).get('hits', 0) >= len(b['answers'])-1 for _, b in pairs)
        result[mode] = {'cache_verified': cache_verified, 'paired_requests': len(pairs), 'binary_disagreements': flips,
                        'acceptance_disagreements': acceptance, 'max_abs_logit_delta': delta,
                        'mean_wall_s': statistics.mean(b['wall_s'] for _, b in pairs),
                        'median_wall_s': statistics.median(b['wall_s'] for _, b in pairs),
                        'speedup': speedup,
                        'peak_allocated_gib': max(b['peak_allocated_gib'] for _, b in pairs),
                        'passed': not flips and not acceptance and cache_verified and speedup >= 1.05}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if 'torch' not in args.modes or len(set(args.modes)) != len(args.modes) or args.repeats < 2:
        parser.error('Require reference torch mode, distinct modes and at least two repeats')
    output = Path(args.output)
    if output.exists(): raise ValueError('Use a fresh report path')
    with gzip.open(args.requests, 'rt') as stream: entries = json.load(stream)
    if not entries: raise ValueError('No requests')
    for entry in entries:
        body = entry['body']
        if request_digest(body['questions'], body['inputs']) != entry['metadata']['request_sha256']:
            raise ValueError('Request integrity check failed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if gpu.strip(): raise RuntimeError('GPU busy; stop model services before replay')
    import torch
    from safetyjev.agentic_server import NativeService
    config = load_config(args.config)
    config['predictor'].update(attention_backend='torch', vision_cache=False, batch_size=1)
    service = NativeService(config)
    report = {'status': 'running', 'samples': [], 'mode_errors': {}, 'warmup_s': {},
              'identity': service.health.copy(), 'repeats': args.repeats,
              'request_file_sha256': hashlib.sha256(Path(args.requests).read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), *sorted((ROOT/'safetyjev').glob('*.py'))]},
              'scope': 'Same loaded checkpoint, batch 1, unchanged masks and LoRA. No simulation or OpenRouter. Timings include input decode and cache hashing; compilation warmup reported separately.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = output.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(output)
    save()
    active = []
    warm = [entries[0]['body'], next(e['body'] for e in entries if all(e['body']['inputs']['history_mask']))]
    for mode in args.modes:
        try:
            start = time.perf_counter(); service.configure_optimizations(*MODES[mode])
            for body in warm: service.predict(body)
            report['warmup_s'][mode] = time.perf_counter()-start; active.append(mode)
        except Exception as exc:
            report['mode_errors'][mode] = type(exc).__name__
            if mode == 'torch': save(); raise
        save()
    for index, entry in enumerate(entries):
        sample = {'metadata': entry['metadata'], 'measurements': []}
        report['samples'].append(sample)
        for repeat in range(args.repeats):
            order = active if (index+repeat)%2 == 0 else list(reversed(active))
            for mode in order:
                service.configure_optimizations(*MODES[mode])
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter(); answer = service.predict(entry['body']); torch.cuda.synchronize()
                sample['measurements'].append({'mode': mode, 'repeat': repeat,
                    'wall_s': time.perf_counter()-start, 'model_region_s': answer['latency_s'],
                    'answers': answer['answers'], 'vision_cache': answer['vision_cache'],
                    'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
                save()
        print(json.dumps({'sample': index+1, 'total': len(entries)}), flush=True)
    report['summary'] = summarize(report['samples'], active)
    report['by_history'] = {phase: summarize([s for s, e in zip(report['samples'], entries)
        if all(e['body']['inputs']['history_mask']) == full], active)
        for phase, full in [('full', True), ('reset_or_partial', False)]}
    eligible = [m for m in active if m != 'torch' and report['summary'][m]['passed']
                and report['by_history']['full'].get(m, {}).get('passed', False)]
    report['selected_mode'] = min(eligible, key=lambda m: report['summary'][m]['mean_wall_s']) if eligible else None
    report['status'] = 'completed'; save()
    print(json.dumps({'summary': report['summary'], 'selected_mode': report['selected_mode'],
                      'mode_errors': report['mode_errors']}, indent=2))


if __name__ == '__main__': main()
