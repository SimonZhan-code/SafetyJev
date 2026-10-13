"""ManiGuard adapter for the separate Codex-controlled experimental arm."""
import base64
import copy
import os
from pathlib import Path
import pwd
import queue
import signal
import subprocess
import time
import json
import threading

import numpy as np

from .agentic import AgenticEpisode
from .agentic_config import validate_config as native_config
from .capture import ShadowEpisode
from .codex_bridge import Bridge
from .codex_config import validate_config, HarnessPredictor
from .codex_registry import Decision, HarnessError
from .io import append_jsonl, write_json


class CodexEpisode(AgenticEpisode):
    execution_mode = 'codex_harness'

    def __init__(self, scene, cfg, monitor, observation, episode_seed, robot):
        from .capture import OPTIONS
        self.harness_config = validate_config(OPTIONS['codex_harness_config'])
        if cfg.max_steps != self.harness_config['max_steps']:
            raise ValueError('ManiGuard cap differs from Codex harness config')
        # Reuse the verified controller/history initialization, never its decision loop.
        previous = OPTIONS.get('agentic_config')
        OPTIONS['agentic_config'] = native_config({'predictor': self.harness_config['predictor'],
                                                 'system1': {'enabled': False}})
        try:
            super().__init__(scene, cfg, monitor, observation, episode_seed, robot)
        finally:
            if previous is None:
                OPTIONS.pop('agentic_config', None)
            else:
                OPTIONS['agentic_config'] = previous
        self.native_predictor = HarnessPredictor(self.harness_config['predictor'])
        self.native_predictor.validate_health(self.provenance['native_predictor'])
        self.bridge = self.worker = None
        self.receipts = {}
        self.current_decision = None
        self.harness_ready = False
        self.log_lock = threading.Lock()
        self.transcript_thread = None
        self.unsafe_selected = self.lower_score_selected = 0
        self.provenance['codex_harness'] = copy.deepcopy(self.harness_config)
        self.provenance['codex_limits'] = ['FLA replay differs on 4/120 binary answers',
            'Codex-written commands are outside validated predictor proposal distribution',
            'Tool and information boundary, not proof of internal model reasoning']
        write_json(self.directory/'episode.json', self.provenance)

    def _log(self, kind, value):
        with self.log_lock:
            append_jsonl(self.directory/'harness-events.jsonl', {'event': kind, **value})
        if kind == 'selection':
            append_jsonl(self.directory/'selections.jsonl', value)

    def _start_worker(self):
        c = self.harness_config
        account = pwd.getpwnam(c['runtime_user'])
        if account.pw_uid == 0:
            raise HarnessError('runtime_must_be_unprivileged')
        path = Path(c['socket_root'])/(self.episode_id+'.sock')
        self.bridge = Bridge(path, account.pw_uid, account.pw_gid)
        env = {'PATH': '/usr/bin:/bin', 'HOME': c['runtime_home'],
               'PYTHONPATH': c['runtime_root'], 'PYTHONNOUSERSITE': '1',
               'CODEX_HOME': c['runtime_home']+'/.codex',
               'LANG': 'C.UTF-8'}
        cmd = [c['runtime_python'], '-m', 'safetyjev.codex_worker', '--socket', str(path),
               '--model', c['model'], '--reasoning-effort', c['reasoning_effort']]
        # No root environment, model credentials, supervisor access, or inherited descriptors.
        self.worker = subprocess.Popen(cmd, env=env, cwd=c['runtime_home']+'/work',
            user=account.pw_uid, group=account.pw_gid, extra_groups=[],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, umask=0o077)
        def transcript():
            for line in self.worker.stdout:
                try:
                    value = json.loads(line)
                    if set(value) == {'event', 'input_tokens', 'output_tokens'} and value['event'] == 'usage':
                        if all(type(value[k]) is int and value[k] >= 0 for k in ('input_tokens', 'output_tokens')):
                            self._log('usage', {k:value[k] for k in ('input_tokens','output_tokens')})
                    elif set(value) == {'event', 'text'} and value['event'] == 'model_text' and isinstance(value['text'], str):
                        self._log('model_text', {'text': value['text'][:65536]})
                except Exception:
                    pass
        self.transcript_thread = threading.Thread(target=transcript, daemon=True)
        self.transcript_thread.start()

    def select_chunk(self, step, observation, initial_chunk, regenerate, action_space, horizon):
        if horizon != 8 or self.native_history.rows[-1]['step'] != step:
            raise HarnessError('invalid_history_alignment')
        c = self.harness_config
        images = {k: base64.b64encode((self.directory/v).read_bytes()).decode()
                  for k, v in self.native_history.rows[-1]['images'].items()}

        def propose(instruction):
            obs = copy.deepcopy(observation)
            obs['task_descriptions'] = instruction
            return regenerate(obs)

        def score(cid, questions, inputs, remaining):
            self.native_predictor.timeout = min(c['predictor']['timeout'], remaining)
            return self.native_predictor.score(cid, questions, inputs)

        d = self.current_decision = Decision(step=step, task=self.original_instruction,
            state=self.native_history.rows[-1]['state'], images=images, history=self.executed_history,
            questions=self.questions, bounds=action_space, propose=propose,
            make_inputs=lambda commands: self.native_history.inputs(commands, 1/self.cfg.action_frequency, self.directory),
            score=score, log=self._log, proposal_limit=c['proposal_limit'], timeout=c['decision_timeout_s'])
        self.decisions += 1
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(*_):
            if d.selected is None:
                raise HarnessError('decision_timeout')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, d.remaining())
        try:
            if self.worker is None:
                self._start_worker()
            while d.selected is None:
                d.check()
                if self.worker.poll() is not None:
                    raise HarnessError('codex_worker_exited')
                try:
                    request, reply = self.bridge.pending.get(timeout=min(.1, d.remaining()))
                except queue.Empty:
                    continue
                method, args = request['method'], request['arguments']
                try:
                    if method == '_runtime_ready':
                        from .codex_worker import validate_attestation
                        validate_attestation(args)
                        self.harness_ready = True
                        self._log('runtime_ready', args)
                        result = {'status': 'ready'}
                    elif method in ('_runtime_failed', '_runtime_done'):
                        raise HarnessError('codex_ended_without_execution')
                    elif method == '_usage':
                        if not isinstance(args, dict) or set(args) != {'input_tokens', 'output_tokens'} or any(type(v) is not int or v < 0 for v in args.values()):
                            raise HarnessError('invalid_usage')
                        self._log('usage', args)
                        result = {'status': 'recorded'}
                    elif not self.harness_ready:
                        raise HarnessError('tool_inventory_not_verified')
                    elif method == 'execute' and isinstance(args, dict) and isinstance(args.get('candidate_id'), str) and args['candidate_id'] in self.receipts:
                        result = self.receipts[args['candidate_id']]
                    else:
                        result = d.dispatch(method, args)
                    if d.selected is not None:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    # This is the complete public response, logged separately from oracle records.
                    self._log('tool', {'tool': method, 'arguments': args, 'response': result})
                    reply.put(result)
                except HarnessError as exc:
                    self._log('tool', {'tool': method, 'arguments': args, 'response': {'error': str(exc)}})
                    reply.put({'error': str(exc)})
                    if d.attempts >= d.limit and not d.pool:
                        raise HarnessError('all_invalid_proposals') from None
                    if str(exc) in ('predictor_failed', 'proposal_failed', 'decision_timeout', 'codex_ended_without_execution', 'tool_inventory_not_verified'):
                        raise
            self._selected = copy.deepcopy(d.selected)
            self.unsafe_selected += int(d.selected['predicted_unsafe_selected'])
            self.lower_score_selected += int(d.selected['higher_score_available'])
            self.receipts[d.selected['candidate_id']] = copy.deepcopy(d.receipt)
            return np.asarray(d.selected['commands'], dtype=np.float32)
        except Exception as exc:
            self.failure_reason = str(exc) if isinstance(exc, HarnessError) else 'harness_failed'
            self._log('failure', {'error': self.failure_reason, 'step': step})
            self._stop_worker()
            raise HarnessError(self.failure_reason) from None
        finally:
            self.candidate_attempts += d.attempts
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def before_action(self, step, observation, chunk, ci, chunk_len, action_space):
        from .agentic_inputs import controller_commands
        if self._selected is None or step != self._selected['start_step'] + ci:
            raise HarnessError('invalid_execution_alignment')
        transformed = controller_commands(chunk, action_space)
        if not np.array_equal(np.asarray(transformed[ci], dtype=np.float32), np.asarray(self._selected['commands'][ci], dtype=np.float32)):
            raise HarnessError('executed_command_changed')

    def after_step(self, step, monitor, observation):
        # AgenticEpisode uses the existing exact-command, post-action-history contract.
        super().after_step(step, monitor, observation)
        if self.ready and self._selected:
            receipt = self.receipts[self._selected['candidate_id']]
            receipt.update(executed_actions=step-self._selected['start_step'], status='executed')

    def _stop_worker(self):
        if self.bridge:
            self.bridge.close()
        if self.worker and self.worker.poll() is None:
            try:
                self.worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.worker.pid, signal.SIGTERM)
                try:
                    self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid, signal.SIGKILL)
                    self.worker.wait()
        if self.transcript_thread:
            self.transcript_thread.join(timeout=2)
        if self.bridge:
            self.bridge.shutdown()
            self.bridge = None

    def finish(self, result):
        self._stop_worker()
        result['safetyjev_codex'] = {'decisions': self.decisions, 'proposal_attempts': self.candidate_attempts,
            'predicted_unsafe_selected': self.unsafe_selected, 'higher_score_available': self.lower_score_selected,
            'infrastructure_failure': self.failure_reason, 'execution_receipts': list(self.receipts.values())}
        if self.failure_reason or self.invalid:
            result.update(status='crashed', success=False)
        ShadowEpisode.finish(self, result)
