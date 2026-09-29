"""Audit the three-mode development capture; does not access any API credentials.

Usage: python audit-planner-integration.py /path/to/artifacts/planner-integration
Assertions check trace consistency, not physics or predictor correctness.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def timing(values):
    return {'count': len(values), 'median_s': statistics.median(values) if values else None,
            'max_s': max(values) if values else None}


def audit(root):
    sweep = json.loads((root/'sweep.json').read_text())
    summary = {'scope': '64-action development integration; not safety improvement or accuracy evidence',
               'cases': [], 'trace_consistency_checks_passed': False}
    initial_inputs = {}
    for case in sweep:
        assert case['status'] == 'passed', case
        episode = root/case['case']/case['episode_id']
        result = json.loads((episode/'maniguard_result.json').read_text())
        oracle = rows(episode/'oracle.jsonl')
        assert [r['step'] for r in oracle] == list(range(result['steps']+1))
        assert all(r['valid'] for r in oracle)
        selected = rows(episode/'forecasts.jsonl')
        predictions = rows(episode/'predictions.jsonl')
        assert {r['forecast_id'] for r in selected} == {r['forecast_id'] for r in predictions}
        assert all(not r.get('error') for r in predictions)
        decisions = rows(episode/'decisions.jsonl')
        accepted = {r['candidate_id']: r for r in decisions if r['decision'] == 'accept'}
        candidates = rows(episode/'candidate_forecasts.jsonl')
        candidate_predictions = rows(episode/'candidate_predictions.jsonl')
        plans = rows(episode/'planner.jsonl')
        initial_inputs[case['case']] = (candidates or selected)[0]['input']
        commands = {}
        if decisions:
            selected_ids = {r['forecast_id'] for r in selected}
            expected_ids = {r['forecast_id'] for r in candidates
                            if r['forecast_id'].rsplit(':', 1)[0] in accepted}
            assert selected_ids == expected_ids, 'Only accepted candidates may receive outcome labels'
            scores = {r['forecast_id']: r for r in candidate_predictions}
            assert len(scores) == len(candidates)
            for decision in decisions:
                fs = [r for r in candidates if r['forecast_id'].rsplit(':',1)[0] == decision['candidate_id']]
                assert len(fs) == 3
                assert all(not scores[f['forecast_id']].get('error') for f in fs)
                passes = all(scores[f['forecast_id']]['score'] < case['threshold'] for f in fs)
                assert passes == (decision['decision'] == 'accept')
                normalized = np.asarray(fs[0]['input']['remaining_actions'],dtype=np.float32)
                assert hashlib.sha256(normalized.tobytes()).hexdigest() == decision['candidate_sha256']
                if passes:
                    for i, action in enumerate(normalized.tolist(), start=fs[0]['start_step']+1):
                        assert i not in commands, 'Overlapping selected windows'
                        commands[i] = (action, decision['policy_instruction'])
            assert sorted(commands) == list(range(1,result['steps']+1)), 'Selected windows must cover executed steps only'
            for plan in plans:
                assert not plan.get('error'), plan
                target = next(r for r in decisions if r['candidate_id'] == plan['candidate_id'])
                assert target['policy_instruction'] == plan['vla_instruction']
                assert any(r['start_step'] == target['start_step'] and r['attempt'] == target['attempt']-1
                           and r['decision'] == 'reject' for r in decisions)
                for frame in plan['input']['images']:
                    assert hashlib.sha256((episode/frame['path']).read_bytes()).hexdigest() == frame['sha256']
                for history in plan['input']['context']['executed_history']:
                    action, instruction = commands[history['step']]
                    assert np.array_equal(np.asarray(history['executed_action'],dtype=np.float32),
                                          np.asarray(action,dtype=np.float32))
                    assert history['policy_instruction'] == instruction
        summary['cases'].append({
            'case': case['case'], 'episode_id': episode.name, 'steps': result['steps'],
            'success': result['success'], 'raw_violation': result['ltl_violated'],
            'raw_first_violation_step': result['ltl_violation_step'],
            'engaged': result['ever_contacted'], 'counted_violation': result['counted_violation'],
            'termination_reason': result.get('safetyjev_guard',{}).get('termination_reason'),
            'candidate_attempts': len(decisions), 'accepted_chunks': len(accepted),
            'rejected_chunks': sum(r['decision']=='reject' for r in decisions),
            'executed_replacement_chunks': sum(r['attempt']>0 for r in accepted.values()),
            'selected_scores': len(predictions), 'candidate_scores': len(candidate_predictions),
            'planner_calls': len(plans), 'planner_latency': timing([r['latency_s'] for r in plans]),
            'planner_reported_cost_usd': sum(r.get('usage',{}).get('cost',0) for r in plans),
            'critic_request_latency': timing([r['latency_s'] for r in candidate_predictions or predictions]),
            'guard_candidate_latency': timing([r['guard_latency_s'] for r in decisions]),
            'process_wall_s': case['ended_unix_s']-case['started_unix_s'],
            'guard_counters': result.get('safetyjev_guard'),
            'oracle_samples': len(oracle),
        })
    with (root/'gpu-memory.csv').open() as f:
        summary['sampled_peak_total_gpu_mib'] = max(float(r['memory_used_mib']) for r in csv.DictReader(f))
    summary['initial_candidate_parity'] = {
        name: {
            'max_absolute_robot_state_delta_from_shadow': float(np.max(np.abs(
                np.asarray(data['robot_state'])-np.asarray(initial_inputs['shadow']['robot_state'])))),
            'max_absolute_action_delta_from_shadow': float(np.max(np.abs(
                np.asarray(data['remaining_actions'])-np.asarray(initial_inputs['shadow']['remaining_actions'])))),
        } for name, data in initial_inputs.items() if name != 'shadow'
    }
    summary['trace_consistency_checks_passed'] = True
    (root/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
