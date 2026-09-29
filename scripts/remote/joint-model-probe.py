"""Synthetic serving probe: no simulator, oracle labels, or accuracy claims."""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from safetyjev.io import write_json
from safetyjev.predictors import OpenJevHTTP


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True)
    p.add_argument('--model-id', required=True, help='Exact identity advertised by Open-Jev /health')
    p.add_argument('--revision', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--diagnostics', required=True)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    fixture = Path(args.fixture)
    policy = WebsocketClientPolicy(host='127.0.0.1', port=8000)
    metadata = policy.get_server_metadata()
    assert metadata['serve_config'] == 'pi05-base_datagen_v1_jar_joint_2cam_lora'
    assert metadata['checkpoint'] == '/workspace/checkpoints/pi05-jar/7400'
    # Read only the static task specification. Historical rollout outcomes are
    # neither assigned to these synthetic actions nor passed to the predictor.
    diagnostic = json.loads(Path(args.diagnostics).read_text().splitlines()[0])
    spec, prompt = diagnostic['ltl_safety'], diagnostic['prompt']
    observation = {
        'observation/state': np.array([0., -.4, 0., -2., 0., 1.6, .8, .04], dtype=np.float32),
        'observation/image_left': np.zeros((256, 256, 3), dtype=np.uint8),
        'observation/wrist_image': np.zeros((256, 256, 3), dtype=np.uint8),
        'prompt': prompt, 'episode_seed': 0,
    }
    policy_times = []
    for _ in range(3):
        start = time.perf_counter()
        actions = np.asarray(policy.infer(dict(observation))['actions'])
        policy_times.append(time.perf_counter() - start)
        assert actions.shape == (16, 8) and np.isfinite(actions).all()
    if not fixture.exists():
        proposals = actions[:8].copy()
        g = proposals[:, -1]
        proposals[:, -1] = np.where(np.abs(g) > .01, np.sign(g), -1.)
        constraints = [dict(c, propositions=spec['propositions']) for c in spec['constraints']]
        constraints.append({'id': '__all__', 'ltl': spec['combined_ltl'],
                            'description': 'Any task safety constraint', 'clauses': constraints.copy()})
        rows = []
        for c in constraints:
            rows.append({'forecast_id': 'synthetic:' + c['id'], 'input': {
                'robot_state': observation['observation/state'].tolist(),
                'past_robot_states': [], 'task_instruction': prompt,
                'remaining_actions': proposals.tolist(), 'action_frequency_hz': 20,
                'action_convention': 'Synthetic serving probe: absolute joint radians(7) + binarized gripper(1); no controller bounds or execution',
                'constraint': c,
            }})
        write_json(fixture, {'synthetic_inputs': True, 'simulator_executed': False, 'forecasts': rows})
    data = json.loads(fixture.read_text())
    predictor = OpenJevHTTP('http://127.0.0.1:8791/v1/systemone', args.model_id, 'proprio_only', 120)
    results = []
    for repeat in range(args.repeats + 1):
        for forecast in data['forecasts']:
            row = predictor.score(forecast)
            row.update(repeat=repeat, warmup=repeat == 0)
            results.append(row)
            print(json.dumps(row), flush=True)
    measured = [r for r in results if not r['warmup']]
    successes = [r for r in measured if r['error'] is None]
    def smi(fields, kind):
        return subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader'], text=True).strip()
    write_json(args.output, {
        'test': 'synthetic_joint_model_serving', 'variant': args.variant, 'revision': args.revision,
        'api_model_id': args.model_id,
        'synthetic_inputs': True, 'simulator_executed': False, 'safety_accuracy_measured': False,
        'input_mode': 'proprio_only', 'policy_metadata': metadata, 'policy_latency_s': policy_times,
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'measured_requests': len(measured), 'successful_requests': len(successes),
        'latency_p50_s': statistics.median(r['latency_s'] for r in successes) if successes else None,
        'gpu_snapshot': smi('name,uuid,memory.total,memory.used,driver_version', 'gpu'),
        'process_snapshot': smi('pid,process_name,used_memory', 'compute-apps'),
        'requests': results,
        'limitations': 'Sequential loopback requests while both models are resident on one GPU. No simulator or labels. No safety-quality inference. Small timing sample with fixed synthetic inputs. GPU memory is a snapshot, not peak.',
    })
    if len(successes) != len(measured):
        raise SystemExit('One or more model requests failed; inspect saved report')


if __name__ == '__main__':
    main()
