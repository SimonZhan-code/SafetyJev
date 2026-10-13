"""Time the pinned Jar policy on recorded robot observations, without simulation."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image


def observation(entry):
    inputs, metadata = entry['body']['inputs'], entry['metadata']
    state = np.asarray(inputs['robot_state'], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError('Expected native 16D robot state')
    images = {}
    for camera in ('overview', 'wrist'):
        encoded = inputs['observations'][camera][-1]
        images[camera] = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))
    # ManiGuard joint-mode policy state is seven arm positions + mean finger position.
    return {'observation/state': state[[0, 1, 2, 3, 4, 5, 6, 14]],
            'observation/image_left': images['overview'],
            'observation/wrist_image': images['wrist'],
            'prompt': metadata['task_instruction'], 'episode_seed': metadata['episode_seed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.repeats < 2:
        parser.error('Use a fresh output and at least two repeats')
    with gzip.open(args.requests, 'rt') as stream:
        entries = json.load(stream)
    if not entries:
        raise ValueError('No replay requests')
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host='127.0.0.1', port=args.port)
    metadata = client.get_server_metadata()
    if (metadata.get('serve_config') != 'pi05-base_datagen_v1_jar_joint_2cam_lora' or
            metadata.get('checkpoint') != '/workspace/checkpoints/pi05-jar/7400'):
        raise ValueError('Unexpected policy identity')
    report = {'status': 'running', 'samples': [], 'server_metadata': metadata,
              'request_sha256': hashlib.sha256(Path(args.requests).read_bytes()).hexdigest(),
              'scope': 'Recorded images and robot proprioception; original task instruction. '
                       'New stochastic proposals, not historical action reproduction. '
                       'No simulator, task outcomes, Codex or safety accuracy measurement.',
              'timing': 'WebSocket round trip; image decoding excluded; warmup excluded from steady mean.'}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output)

    start = time.perf_counter()
    warm = client.infer(observation(entries[0]))
    report['warmup_s'] = time.perf_counter() - start
    if np.asarray(warm['actions']).shape != (16, 8):
        raise ValueError('Unexpected warmup action shape')
    for entry in entries:
        measurements = []
        for repeat in range(args.repeats):
            obs = observation(entry)
            start = time.perf_counter()
            response = client.infer(obs)
            elapsed = time.perf_counter() - start
            actions = np.asarray(response['actions'])
            if actions.shape != (16, 8) or not np.isfinite(actions).all():
                raise ValueError('Invalid policy action chunk')
            measurements.append({'repeat': repeat, 'wall_s': elapsed,
                                 'shape': list(actions.shape), 'all_finite': True,
                                 'raw_actions': actions.tolist()})
        report['samples'].append({'metadata': entry['metadata'], 'measurements': measurements})
        save()
        print(json.dumps({'scene': entry['metadata']['scene'], 'step': entry['metadata']['step'],
                          'mean_s': statistics.mean(x['wall_s'] for x in measurements)}), flush=True)
    times = [m['wall_s'] for s in report['samples'] for m in s['measurements']]
    report.update(status='completed', mean_wall_s=statistics.mean(times),
                  median_wall_s=statistics.median(times), measured_requests=len(times))
    save()


if __name__ == '__main__':
    main()
