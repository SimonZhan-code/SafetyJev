"""Trusted decision state. No simulator objects or oracle values cross this API."""
import copy
import hashlib
import json
import time
import uuid

import numpy as np

from .agentic_inputs import controller_commands
from .agentic_predictor import request_digest

TOOLS = ('observe', 'propose_pi05', 'propose_actions', 'judge', 'execute')


class HarnessError(RuntimeError):
    """Messages are fixed error codes, never exception text from dependencies."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Decision:
    def __init__(self, *, step, task, state, images, history, questions, bounds,
                 propose, make_inputs, score, log, proposal_limit=3, timeout=120,
                 clock=time.monotonic, report_budget=False, soft_target=None):
        self.report_budget = report_budget
        self.soft_target = soft_target
        self.prepared_candidate = None
        self.preparation_started = False
        self.id = uuid.uuid4().hex
        self.step, self.clock = step, clock
        self.started = clock()
        self.deadline = self.started + timeout
        self.limit, self.attempts = proposal_limit, 0
        self.questions = [{'id': q['id'], 'question': q['question']} for q in questions]
        if not self.questions or len({q['id'] for q in self.questions}) != len(self.questions):
            raise HarnessError('invalid_constraints')
        if np.asarray(state).shape != (16,) or not np.isfinite(state).all():
            raise HarnessError('invalid_robot_state')
        self.observation = {
            'task': copy.deepcopy(task), 'robot_state': list(state),
            'images': {k: images[k] for k in ('overview', 'wrist')},
            'executed_history': [{k: copy.deepcopy(r[k]) for k in
                                  ('step', 'robot_state', 'executed_action')} for r in history[-8:]],
            'constraints': copy.deepcopy(self.questions),
            'action_convention': '8 actions; each is 7 absolute joint radians plus binary gripper',
            'controller_bounds': {k: [float(v) if np.isfinite(v) else None for v in getattr(bounds, k)]
                                  for k in ('low', 'high')},
        }
        self.bounds, self.propose = bounds, propose
        self.make_inputs, self.score, self.log = make_inputs, score, log
        self.pool, self.cache = {}, {}
        self.selected, self.receipt, self.closed = None, None, False

    def remaining(self):
        return max(0., self.deadline - self.clock())

    def budget(self):
        return {'seconds_remaining': self.remaining(),
                'proposals_remaining': self.limit-self.attempts} if self.report_budget else {}

    def check(self):
        if self.closed or self.selected is not None:
            raise HarnessError('decision_closed')
        if self.remaining() <= 0:
            self.closed = True
            raise HarnessError('decision_timeout')

    def prepare(self, instruction):
        # Trusted runner only; never exposed as an MCP tool. One attempt per state.
        self.check()
        if self.preparation_started:
            raise HarnessError('already_prepared')
        self.preparation_started = True
        started = self.clock()
        proposal = self.proposal(self.id, source='pi05', instruction=instruction)
        prediction = self.judge(proposal['candidate_id'])
        self.prepared_candidate = {'proposal': proposal, 'prediction': prediction}
        self.log('preparation', {'candidate_id': proposal['candidate_id'],
                               'latency_s': self.clock()-started})

    def observe(self):
        self.check()
        return {**copy.deepcopy(self.observation), 'decision_id': self.id,
                'step': self.step, 'proposals_remaining': self.limit-self.attempts,
                'seconds_remaining': self.remaining(),
                **({'prepared_candidate': {k: {**copy.deepcopy(v), **self.budget()}
                    for k, v in self.prepared_candidate.items()}} if self.prepared_candidate else {})}

    def proposal(self, decision_id, *, source, instruction=None, actions=None):
        self.check()
        if decision_id != self.id:
            raise HarnessError('stale_decision')
        if self.attempts >= self.limit:
            raise HarnessError('proposal_budget_exhausted')
        self.attempts += 1
        started = self.clock()
        cid = uuid.uuid4().hex
        try:
            if source == 'pi05':
                if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 512:
                    raise HarnessError('invalid_instruction')
                actions = self.propose(instruction)
                if np.asarray(actions).ndim == 2:
                    actions = actions[:8]
            elif source not in ('codex', 'joint_primitive'):
                raise HarnessError('invalid_source')
            if np.asarray(actions).shape != (8, 8):
                raise HarnessError('invalid_actions')
            commands = controller_commands(actions, self.bounds).astype(np.float32).tolist()
            self.check()
        except Exception as exc:
            code = str(exc) if isinstance(exc, HarnessError) else 'proposal_failed'
            self.log('proposal', {'candidate_id': cid, 'source': source, 'valid': False,
                                 'error': code, 'attempt': self.attempts})
            raise HarnessError(code) from None
        candidate = {'candidate_id': cid, 'decision_id': self.id, 'start_step': self.step,
                     'source': source, 'commands': commands, 'commands_sha256': digest(commands),
                     'policy_instruction': instruction if source == 'pi05' else None,
                     'answers': None, 'generation_latency_s': self.clock()-started}
        self.pool[cid] = candidate
        self.log('proposal', {**copy.deepcopy(candidate), 'valid': True, 'attempt': self.attempts})
        result = {k: copy.deepcopy(candidate[k]) for k in
                  ('candidate_id', 'decision_id', 'source', 'commands', 'commands_sha256', 'policy_instruction')}
        return {**result, **self.budget()}

    def candidate(self, cid):
        self.check()
        if not isinstance(cid, str) or cid not in self.pool:
            raise HarnessError('unknown_candidate')
        c = self.pool[cid]
        if c['decision_id'] != self.id or digest(c['commands']) != c['commands_sha256']:
            raise HarnessError('candidate_changed')
        return c

    def judge(self, cid):
        c = self.candidate(cid)
        started = self.clock()
        try:
            inputs = self.make_inputs(copy.deepcopy(c['commands']))
            key = request_digest(self.questions, inputs)
            cached = key in self.cache
            answers = dict(self.cache[key]) if cached else self.score(cid, self.questions, inputs, self.remaining())
            self.check()
            if set(answers) != {q['id'] for q in self.questions} or any(type(v) is not bool for v in answers.values()):
                raise HarnessError('incomplete_prediction')
        except Exception as exc:
            c['answers'] = None
            code = str(exc) if isinstance(exc, HarnessError) else 'predictor_failed'
            self.log('judgment', {'candidate_id': cid, 'valid': False, 'error': code})
            raise HarnessError(code) from None
        self.cache[key] = dict(answers)
        c.update(answers=dict(answers), prediction_id=key, satisfied_constraints=sum(not v for v in answers.values()))
        result = {'candidate_id': cid, 'prediction_id': key,
                  'answers': {k: 'Yes' if v else 'No' for k, v in answers.items()},
                  'satisfied_constraints': c['satisfied_constraints']}
        self.log('judgment', {**result, 'valid': True, 'cached': cached,
                              'latency_s': self.clock()-started})
        return {**result, **self.budget()}

    def execute(self, cid, rationale):
        # Candidate ID is also the execution idempotency key. Receipts survive closure.
        if self.receipt and cid == self.receipt['candidate_id']:
            return copy.deepcopy(self.receipt)
        c = self.candidate(cid)
        if c['answers'] is None:
            raise HarnessError('prediction_required')
        if not isinstance(rationale, dict) or set(rationale) != {'prediction_id', 'constraint_answers', 'text'}:
            raise HarnessError('invalid_rationale')
        expected = {k: 'Yes' if v else 'No' for k, v in c['answers'].items()}
        if (rationale['prediction_id'] != c['prediction_id'] or rationale['constraint_answers'] != expected
                or not isinstance(rationale['text'], str) or not 1 <= len(rationale['text']) <= 2048):
            raise HarnessError('invalid_predictor_citation')
        self.check()
        chosen = copy.deepcopy(c)
        chosen.update(predicted_unsafe_selected=any(c['answers'].values()),
                      higher_score_available=any(x.get('satisfied_constraints', -1) > c['satisfied_constraints']
                                                 for x in self.pool.values()),
                      rationale=copy.deepcopy(rationale), selection_latency_s=self.clock()-self.started)
        if self.soft_target is not None:
            chosen['soft_target_exceeded'] = chosen['selection_latency_s'] > self.soft_target
        receipt = {'status': 'committed', 'candidate_id': cid, 'execution_id': cid,
                   'decision_id': self.id, 'commands_sha256': c['commands_sha256'], 'planned_actions': 8}
        # Persist before committing. A failed log write never authorizes execution.
        self.log('selection', chosen)
        self.check()
        self.selected, self.receipt, self.closed = chosen, receipt, True
        return copy.deepcopy(receipt)

    def dispatch(self, tool, args):
        schemas = {'observe': set(), 'propose_pi05': {'decision_id', 'instruction'},
                   'propose_actions': {'decision_id', 'actions'}, 'judge': {'candidate_id'},
                   'execute': {'candidate_id', 'rationale'}}
        if tool not in schemas or not isinstance(args, dict) or set(args) != schemas[tool]:
            # Count malformed proposal submissions against the current decision too.
            if tool in ('propose_pi05', 'propose_actions'):
                self.check()
                if self.attempts < self.limit:
                    self.attempts += 1
                    self.log('proposal', {'source': 'pi05' if tool == 'propose_pi05' else 'codex',
                        'valid': False, 'error': 'invalid_tool_request', 'attempt': self.attempts})
            raise HarnessError('invalid_tool_request')
        if tool == 'observe':
            return self.observe()
        if tool == 'propose_pi05':
            return self.proposal(args['decision_id'], source='pi05', instruction=args['instruction'])
        if tool == 'propose_actions':
            return self.proposal(args['decision_id'], source='codex', actions=args['actions'])
        if tool == 'judge':
            return self.judge(args['candidate_id'])
        return self.execute(args['candidate_id'], args['rationale'])
