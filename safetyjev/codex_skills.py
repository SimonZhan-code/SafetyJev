"""Bounded, predictor-gated skills. Only the simulator's owner thread advances them.

Delegation uses the explicitly configured selection policy. A
warning is evidence from the model, not an oracle label or proof of danger.
"""
import copy
from contextlib import contextmanager, nullcontext
import queue
import signal
import time

import numpy as np

from .codex_registry import HarnessError, digest


def validate_skill(args, limit):
    fields = {'request_id', 'decision_id', 'kind', 'instruction', 'joint_positions', 'gripper', 'max_chunks'}
    if not isinstance(args, dict) or set(args) != fields:
        raise HarnessError('invalid_skill_request')
    if any(not isinstance(args[k], str) or not 1 <= len(args[k]) <= 128 for k in ('request_id', 'decision_id')):
        raise HarnessError('invalid_skill_request')
    if type(args['max_chunks']) is not int or not 1 <= args['max_chunks'] <= limit:
        raise HarnessError('invalid_skill_bound')
    kind = args['kind']
    if kind == 'vla':
        if (not isinstance(args['instruction'], str) or not 1 <= len(args['instruction'].strip()) <= 512
                or args['joint_positions'] != [] or args['gripper'] is not None):
            raise HarnessError('invalid_skill_request')
    elif kind in ('hold_joints', 'joint_target'):
        if args['instruction'] != '' or type(args['gripper']) not in (int, float) or args['gripper'] not in (-1, 1):
            raise HarnessError('invalid_skill_request')
        joints = args['joint_positions']
        if not isinstance(joints, list) or len(joints) != (7 if kind == 'joint_target' else 0):
            raise HarnessError('invalid_skill_request')
        if any(type(x) not in (int, float) or not np.isfinite(x) for x in joints):
            raise HarnessError('invalid_skill_request')
    else:
        raise HarnessError('unknown_skill')
    return copy.deepcopy(args)


def primitive_commands(spec, state, delta_limit):
    """Use robot joints only; gripper width in proprioception is NOT a command."""
    q = np.asarray(state[:7], dtype=np.float64)
    target = q if spec['kind'] == 'hold_joints' else np.asarray(spec['joint_positions'])
    if np.any(np.abs(target-q) > delta_limit + 1e-8):
        raise HarnessError('joint_target_out_of_range')
    commands = np.empty((8, 8), dtype=np.float64)
    commands[:, :7] = q + np.arange(1, 9)[:, None]/8 * (target-q)
    commands[:, 7] = spec['gripper']
    return commands


def guarded_choice(decision, spec, delta_limit, commit_guard=nullcontext, selection_policy="all_clear"):
    """One VLA resample on warning; no redundant retry of a fixed primitive."""
    if selection_policy not in ('all_clear', 'least_violations'):
        raise HarnessError('invalid_selection_policy')
    predictions = []
    for _ in range(2 if spec['kind'] == 'vla' else 1):
        if spec['kind'] == 'vla':
            proposal = decision.proposal(decision.id, source='pi05', instruction=spec['instruction'])
        else:
            proposal = decision.proposal(decision.id, source='joint_primitive',
                actions=primitive_commands(spec, decision.observation['robot_state'], delta_limit))
        prediction = decision.judge(proposal['candidate_id'])
        predictions.append({k: copy.deepcopy(prediction[k]) for k in
                            ('candidate_id', 'prediction_id', 'answers', 'satisfied_constraints')})
        if all(v == 'No' for v in prediction['answers'].values()):
            with commit_guard():
                decision.execute(proposal['candidate_id'], {
                    'prediction_id': prediction['prediction_id'], 'constraint_answers': prediction['answers'],
                    'text': 'Delegated skill '+spec['request_id']+': all fixed predictor answers are No.'})
            return predictions
    if selection_policy == 'least_violations':
        # Stable max chooses the earliest candidate on a tie. All candidates here
        # have complete, successful predictions for this exact immutable state.
        best = max(predictions, key=lambda p: p['satisfied_constraints'])
        with commit_guard():
            decision.execute(best['candidate_id'], {
                'prediction_id': best['prediction_id'], 'constraint_answers': best['answers'],
                'text': 'Delegated skill '+spec['request_id']+': least predicted violations after bounded retry; earliest tie.'})
    return predictions


class SkillLoop:
    def __init__(self, episode, clock=time.monotonic):
        self.e, self.clock = episode, clock
        self.active = None
        self.requests, self.completed = {}, {}
        self.calls = self.blocked = self.local_retries = 0
        self.blocked_step, self.blocked_at_step = None, 0
        self.policy_stop = None
        self.warning_cache = {}
        self.prediction_cache = {}
        self.cache_step = None
        self.wait_started = None

    @contextmanager
    def _commit_guard(self):
        # Cancellation and commitment have an explicit order. A cancellation that
        # wins this lock prevents the next chunk; an already committed chunk finishes.
        with getattr(self.e.bridge, 'commit_lock', nullcontext()):
            cancel = getattr(self.e.bridge, 'cancel_requested', None)
            if cancel is not None and cancel.is_set():
                raise HarnessError('skill_cancelled')
            if self.e.worker.poll() is not None or getattr(self.e.bridge, 'closed', False):
                raise HarnessError('codex_worker_exited')
            yield

    def _reply(self, method, args, reply, result):
        self.e._log('tool', {'tool': method, 'arguments': args, 'response': result})
        reply.put(copy.deepcopy(result))

    def _close_skill(self, step, status, predictions=None, terminal=False):
        a = self.active
        if not a:
            return
        public = {'status': 'session_closed'} if terminal else {
            'request_id': a['spec']['request_id'], 'status': status,
            'executed_actions': step-a['start_step'], 'chunks_committed': a['chunks'],
            'predictor_evidence': a['committed_evidence'] if predictions is None else predictions,
            'elapsed_s': self.clock()-a['started']}
        self.e._log('skill_end', {'request_id': a['spec']['request_id'], 'status': status,
            'executed_actions': step-a['start_step'], 'chunks_committed': a['chunks'],
            'elapsed_s': self.clock()-a['started'], 'step': step})
        self.completed[a['spec']['request_id']] = copy.deepcopy(public)
        self._reply('run_skill', a['spec'], a['reply'], public)
        self.active = None
        self.wait_started = self.clock()

    def _stop(self, reason, step):
        self.policy_stop = reason
        self.e._log('policy_stop', {'reason': reason, 'step': step})
        return None

    def _observation(self, d):
        obs = d.observe()
        obs['skill_budget'] = {'max_chunks': self.e.harness_config['skill_max_chunks'],
            'calls_remaining': max(0, self.e.harness_config['skill_call_limit']-self.calls),
            'replans_remaining': max(0, 1+self.e.harness_config['skill_replans_without_progress']-
                (self.blocked_at_step if self.blocked_step == d.step else 0)),
            'joint_delta_limit': self.e.harness_config['skill_joint_delta_limit']}
        return obs

    def select_chunk(self, step, observation, regenerate, bounds, horizon):
        e, c = self.e, self.e.harness_config
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(*_):
            raise HarnessError('skill_or_planner_timeout')
        signal.signal(signal.SIGALRM, timeout)
        try:
            if self.cache_step != step:
                self.warning_cache.clear(); self.prediction_cache.clear(); self.cache_step = step
            if e.worker is None:
                signal.setitimer(signal.ITIMER_REAL, c['decision_timeout_s'])
                e._start_worker()
            if self.wait_started is None:
                self.wait_started = self.clock()
            d = None
            while True:
                cancel = getattr(e.bridge, 'cancel_requested', None)
                if cancel is not None and cancel.is_set():
                    raise HarnessError('skill_cancelled')
                if e.worker.poll() is not None or getattr(e.bridge, 'closed', False):
                    raise HarnessError('codex_worker_exited')
                if self.active:
                    a = self.active
                    if a['chunks'] >= a['spec']['max_chunks']:
                        self._close_skill(step, 'chunk_budget')
                        d = None
                        continue
                    remaining = a['started'] + c['decision_timeout_s'] - self.clock()
                    if remaining <= 0:
                        raise HarnessError('skill_or_planner_timeout')
                    signal.setitimer(signal.ITIMER_REAL, remaining)
                    d = e._new_decision(step, observation, regenerate, bounds, horizon)
                    d.deadline = min(d.deadline, d.started+remaining)
                    d.cache = self.prediction_cache
                    e.decisions += 1
                    e._log('skill_chunk', {'request_id': a['spec']['request_id'],
                        'decision_id': d.id, 'step': step})
                    try:
                        predictions = guarded_choice(d, a['spec'], c['skill_joint_delta_limit'], self._commit_guard, c['skill_selection_policy'])
                    finally:
                        e.candidate_attempts += d.attempts
                    self.local_retries += max(0, d.attempts-1)
                    if d.selected is not None:
                        a['chunks'] += 1
                        a['committed_evidence'].append(copy.deepcopy(next(p for p in predictions if p['candidate_id']==d.selected['candidate_id'])))
                        e._selected = copy.deepcopy(d.selected)
                        e.receipts[d.selected['candidate_id']] = copy.deepcopy(d.receipt)
                        # The committed eight actions finish atomically under the existing runner.
                        return np.asarray(d.selected['commands'], dtype=np.float32)
                    self.blocked += 1
                    self.blocked_at_step = self.blocked_at_step+1 if self.blocked_step == step else 1
                    self.blocked_step = step
                    key = digest({k:v for k,v in a['spec'].items() if k not in ('request_id','decision_id','max_chunks')})
                    self.warning_cache[key] = copy.deepcopy(predictions)
                    terminal = self.blocked_at_step > c['skill_replans_without_progress']
                    self._close_skill(step, 'predictor_warning', predictions, terminal=terminal)
                    if terminal:
                        return self._stop('predictor_blocked', step)
                    d = None
                    continue
                if self.calls >= c['skill_call_limit']:
                    return self._stop('planner_call_budget', step)
                remaining = self.wait_started+c['decision_timeout_s']-self.clock()
                if remaining <= 0:
                    raise HarnessError('skill_or_planner_timeout')
                signal.setitimer(signal.ITIMER_REAL, remaining)
                if d is None:
                    d = e._new_decision(step, observation, regenerate, bounds, horizon)
                    d.deadline = min(d.deadline, d.started+remaining)
                try:
                    request, reply = e.bridge.pending.get(timeout=min(.1, remaining))
                except queue.Empty:
                    continue
                method, args = request['method'], request['arguments']
                try:
                    if method == '_runtime_ready':
                        from .codex_worker import validate_attestation
                        validate_attestation(args, 'skills')
                        e.harness_ready = True
                        self._reply(method, args, reply, {'status': 'ready'})
                    elif method in ('_runtime_failed', '_runtime_done'):
                        raise HarnessError('codex_ended_without_execution')
                    elif not e.harness_ready:
                        raise HarnessError('tool_inventory_not_verified')
                    elif method == 'observe' and args == {}:
                        self._reply(method, args, reply, self._observation(d))
                    elif method == 'run_skill':
                        # Replaying an exact old request only replays its receipt, never actions.
                        rid = args.get('request_id') if isinstance(args, dict) else None
                        if isinstance(rid, str) and rid in self.requests:
                            if args != self.requests[rid]:
                                raise HarnessError('request_id_conflict')
                            self._reply(method, args, reply, self.completed[rid])
                            continue
                        self.calls += 1  # Invalid requests also consume the bounded planner budget.
                        spec = validate_skill(args, c['skill_max_chunks'])
                        if spec['decision_id'] != d.id:
                            raise HarnessError('stale_decision')
                        if spec['kind'] != 'vla':
                            primitive_commands(spec, d.observation['robot_state'], c['skill_joint_delta_limit'])
                        self.requests[rid] = copy.deepcopy(spec)
                        key = digest({k:v for k,v in spec.items() if k not in ('request_id','decision_id','max_chunks')})
                        self.active = {'spec': spec, 'reply': reply, 'start_step': step,
                            'started': self.clock(), 'chunks': 0, 'committed_evidence': []}
                        e._log('skill_start', {'spec': spec, 'step': step,
                            'planner_wait_s': self.clock()-self.wait_started, 'selection_policy': c['skill_selection_policy']})
                        if key in self.warning_cache:
                            # No progress and no changed plan: don't pay for identical inference again.
                            self.blocked_at_step += 1
                            self.blocked += 1
                            terminal = self.blocked_at_step > c['skill_replans_without_progress']
                            self._close_skill(step, 'predictor_warning', self.warning_cache[key], terminal)
                            if terminal:
                                return self._stop('predictor_blocked', step)
                        d = None
                    else:
                        raise HarnessError('invalid_tool_request')
                except HarnessError as exc:
                    self._reply(method, args, reply, {'error': str(exc)})
                    if str(exc) in ('codex_ended_without_execution','tool_inventory_not_verified'):
                        raise
        except Exception as exc:
            e.failure_reason = str(exc) if isinstance(exc, HarnessError) else 'skill_harness_failed'
            e._log('failure', {'error': e.failure_reason, 'step': step})
            self._close_skill(step, 'infrastructure_failure', terminal=True)
            e._stop_worker()
            raise HarnessError(e.failure_reason) from None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def finish(self, step):
        # Early simulator termination never sends its private reason to Codex.
        self._close_skill(step, 'session_closed', terminal=True)

    def metrics(self):
        return {'skill_requests': self.calls, 'warning_returns': self.blocked,
                'local_retries': self.local_retries, 'policy_stop': self.policy_stop,
                'selection_policy': ('least_predicted_violations_earliest_tie' if self.e.harness_config['skill_selection_policy']=='least_violations' else 'all_constraints_No_or_pause')}
