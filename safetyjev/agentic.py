"""Binary native-predictor selection with bounded local and OpenRouter repair."""
import copy
import time
import numpy as np
from .agentic_config import public_config, validate_config
from .agentic_inputs import ChunkHistory, bind_queries, controller_commands, robot_state
from .agentic_predictor import NativePredictor, request_digest
from .agentic_progress import ProgressWindow
from .capture import ShadowEpisode
from .io import append_jsonl, write_json
from .planner import OpenRouterPlanner


class AgenticInfrastructureError(RuntimeError):
    pass


class AgenticEpisode(ShadowEpisode):
    execution_mode = 'agentic_native_predictor'

    def __init__(self, scene, cfg, monitor, observation, episode_seed, robot):
        from .capture import OPTIONS
        self.config = validate_config(OPTIONS['agentic_config'])
        if (cfg.execute_horizon != 8 or cfg.max_steps <= 0 or cfg.max_steps % 8 or
                not cfg.gripper_binarize):
            raise ValueError('Native agentic mode requires horizon 8, an action cap divisible by 8, and binary gripper commands')
        arm = robot.controllers['arm_'+robot.default_arm]
        grip = robot.controllers['gripper_'+robot.default_arm]
        if (type(arm).__name__ != 'JointController' or arm._motor_type != 'position' or arm._use_delta_commands or
                arm._command_input_limits is not None or arm._command_output_limits is not None or
                arm._use_impedances or arm.command_dim != 7 or robot._action_normalize or
                type(grip).__name__ != 'MultiFingerGripperController' or grip._mode != 'binary' or
                grip._command_input_limits is not None or grip.command_dim != 1):
            raise ValueError('Native checkpoint requires its trained absolute joint controller')
        self.ready = False
        self.robot = robot
        self.native_history = ChunkHistory()
        self.executed_history = []
        self._selected = None
        self.planner_calls = self.fallbacks = self.decisions = self.candidate_attempts = 0
        self.failure_reason = None
        self._initialize_recovery(observation)
        super().__init__(scene, cfg, monitor, observation, episode_seed)
        self.queries = bind_queries(scene, monitor._resolver,
                                   {ap for m in self.machines.values() for ap in m.ap_list}, self.config['constraint_ids'])
        self.questions = [{'id': q['id'], 'question': q['question']} for q in self.queries]
        self.native_predictor = NativePredictor(self.config['predictor'])
        self.planner = None
        system1 = self.config['system1']
        if system1['enabled'] and system1['max_calls_per_episode'] and self.config['repairs_per_decision']:
            self.planner = OpenRouterPlanner({k:v for k,v in system1.items() if k != 'enabled'})
        self.provenance.update(agentic=public_config(self.config), native_predictor=self.native_predictor.health(),
            classification_queries=self.queries,
            label_semantics='Native scores predict trained semantic violations, including ongoing violations. '
                            'oracle.jsonl contains separate ManiGuard AP/LTL outcomes, not native semantic labels.')
        write_json(self.directory/'episode.json', self.provenance)
        self._observe(0, observation)
        self.ready = True

    def _initialize_recovery(self, observation):
        self.original_instruction = copy.deepcopy(observation['task_descriptions'])
        self.active_instruction = None
        self.progress_window = ProgressWindow(self.config['stagnation'])
        self.progress_report = {'ready': False, 'stalled': False}
        self.last_planner_step = None
        self._prepared_step = None
        self._stagnation_due = False

    def _repair_available(self):
        return (self.config['system1']['enabled'] and self.planner is not None and
                self.config['repairs_per_decision'] > 0 and
                self.planner_calls < self.config['system1']['max_calls_per_episode'])

    def _clear_instruction(self, step, reason):
        if self.active_instruction is not None:
            append_jsonl(self.directory/'recovery.jsonl',
                         {'step': step, 'event': 'instruction_cleared', 'reason': reason,
                          'active_instruction': self.active_instruction})
            self.active_instruction = None

    def policy_observation(self, step, observation):
        """Prepare the initial VLA prompt once per executed chunk boundary."""
        if self.native_history.rows[-1]['step'] != step:
            raise AgenticInfrastructureError('Recovery history does not match decision state')
        if self._prepared_step != step:
            self._prepared_step = step
            if self.active_instruction and step >= self.active_instruction['expires_step']:
                self._clear_instruction(step, 'ttl_expired')
            self._stagnation_due = False
            if self.config['stagnation']['enabled']:
                self.progress_report = self.progress_window.observe(
                    step, self.native_history.rows[-1]['state'], observation['overview_image'])
                cooldown = self.config['stagnation']['cooldown_chunks'] * 8
                cooled = self.last_planner_step is None or step - self.last_planner_step >= cooldown
                self._stagnation_due = self.progress_report['stalled'] and cooled and self._repair_available()
                append_jsonl(self.directory/'recovery.jsonl', {'step': step, 'event': 'progress',
                    **self.progress_report, 'repair_due': self._stagnation_due,
                    'cooldown_ready': cooled, 'repair_available': self._repair_available()})
                if self._stagnation_due:
                    self._clear_instruction(step, 'stagnation')
        result = copy.deepcopy(observation)
        result['task_descriptions'] = (self.active_instruction['instruction'] if self.active_instruction
                                       else copy.deepcopy(self.original_instruction))
        return result

    def _observe(self, step, observation, command=None):
        images = {}
        for camera, key in [('overview','overview_image'),('wrist','wrist_images')]:
            name = f'frames/{step:07d}-{camera}.png'
            self.imageio.imwrite(self.directory/name, observation[key])
            images[camera] = name
        state = robot_state(self.robot)
        self.native_history.append(step, state, images, command)
        append_jsonl(self.directory/'native-observations.jsonl',
                     {'step':step,'robot_state':state,'images':images,'executed_action':command})

    def before_action(self, *args):
        pass

    def _context(self, step, candidate_history, commands, trigger):
        current = self.native_history.rows[-1]
        context = {'original_task': self.original_instruction,
            'current_instruction': candidate_history[-1]['policy_instruction'], 'start_step':step,
            'robot_state':current['state'], 'remaining_actions':commands,
            'action_convention':'absolute joint radians(7) + binary gripper(1), clipped to controller bounds',
            'action_frequency_hz':self.cfg.action_frequency,
            'constraints':[{'id':q['id'],'description':q['question']} for q in self.queries],
            'guard_feedback':candidate_history[-1]['guard_feedback'],
            'candidate_history':copy.deepcopy(candidate_history), 'executed_history':copy.deepcopy(self.executed_history),
            'images':{'overview_image':current['images']['overview'],'wrist_images':current['images']['wrist']}}

        if self.config['instruction_ttl_chunks'] > 1 or self.config['stagnation']['enabled']:
            context.update(repair_trigger=trigger, instruction_ttl_chunks=self.config['instruction_ttl_chunks'],
                           progress=copy.deepcopy(self.progress_report))
        return context

    def select_chunk(self, step, observation, initial_chunk, regenerate, action_space, horizon):
        if horizon != 8:
            raise AgenticInfrastructureError('Native predictor requires eight committed actions')
        if self.native_history.rows[-1]['step'] != step:
            raise AgenticInfrastructureError('Native history does not match decision state')
        self._selected = None
        self.decisions += 1
        started = time.monotonic()
        pool, cache, candidate_history = [], {}, []
        candidate_obs = self.policy_observation(step, observation)
        stagnation_due = self._stagnation_due
        chunk, attempt, last_commands = initial_chunk, 0, []
        reason = 'repair_budget_exhausted'
        for repair in range(self.config['repairs_per_decision']+1):
            for local in range(self.config['local_retries']+1):
                candidate_id = f'{self.episode_id}:{step}:candidate-{attempt}'
                generation_started = time.monotonic()
                if attempt:
                    try:
                        chunk = regenerate(copy.deepcopy(candidate_obs))
                    except Exception:
                        self.failure_reason = 'policy_error'
                        append_jsonl(self.directory/'selections.jsonl', {'step':step,'error':'policy_error'})
                        raise AgenticInfrastructureError('VLA proposal failed') from None
                generation_s = time.monotonic()-generation_started if attempt else None
                attempt += 1
                self.candidate_attempts += 1
                answers, error, digest, commands = {}, None, None, None
                duplicate = False
                scoring_started = time.monotonic()
                try:
                    commands = controller_commands(chunk, action_space)
                    inputs = self.native_history.inputs(commands, 1/self.cfg.action_frequency, self.directory)
                    digest = request_digest(self.questions, inputs)
                    append_jsonl(self.directory/'candidate-inputs.jsonl', {
                        'candidate_id':candidate_id,'step':step,'request_sha256':digest,
                        'inputs':{k:v for k,v in inputs.items() if k != 'observations'},
                        'questions':self.questions,
                        'history_images':[{'step':r['step'],'images':r['images']} for r in self.native_history.rows]})
                    duplicate = digest in cache
                    if duplicate:
                        answers = dict(cache[digest])
                    else:
                        answers = self.native_predictor.score(candidate_id, self.questions, inputs)
                        if set(answers) != {q['id'] for q in self.questions} or any(type(v) is not bool for v in answers.values()):
                            raise ValueError('Incomplete native answer set')
                        cache[digest] = dict(answers)
                except Exception as exc:
                    answers, error = {}, type(exc).__name__
                valid = bool(answers) and error is None
                score = sum(not value for value in answers.values()) if valid else None
                accepted = valid and score == len(self.questions)
                entry = {'candidate_id':candidate_id,'step':step,'repair_round':repair,'local_attempt':local,
                    'policy_instruction':candidate_obs['task_descriptions'], 'violations':answers,
                    'satisfied_constraints':score,'valid':valid,'decision':'accept' if accepted else 'reject',
                    'error':error,'duplicate':duplicate,'request_sha256':digest,
                    'generation_latency_s':generation_s,'prediction_latency_s':time.monotonic()-scoring_started}
                append_jsonl(self.directory/'candidate-decisions.jsonl',entry)
                feedback = [{'constraint_id':q['id'], 'verdict':('violated' if answers[q['id']] else 'no_violation')
                             if q['id'] in answers else 'unknown'} for q in self.questions]
                candidate_history.append({'candidate_id':candidate_id,'start_step':step,
                    'policy_instruction':candidate_obs['task_descriptions'],'decision':entry['decision'],
                    'guard_feedback':feedback})
                last_commands = commands.tolist() if commands is not None else []
                if valid:
                    candidate = {**entry,'commands':commands.tolist(),'start_step':step}
                    pool.append(candidate)
                    if accepted:
                        if stagnation_due and repair == 0:
                            # Retain this safe option while trying an instruction that may make progress.
                            break
                        return self._select(candidate, False, 'all_constraints_pass', started)
            if repair == self.config['repairs_per_decision']:
                break
            if not self._repair_available():
                reason = 'planner_disabled_or_episode_budget_exhausted'
                break
            trigger = 'stagnation' if stagnation_due and repair == 0 else 'safety_rejection'
            self.planner_calls += 1
            self.last_planner_step = step
            planner_started = time.monotonic()
            try:
                plan = self.planner.plan(self._context(step, candidate_history, last_commands, trigger), self.directory)
                append_jsonl(self.directory/'planner.jsonl', {'step':step,'repair_round':repair+1,'trigger':trigger,**plan})
                candidate_obs['task_descriptions'] = plan['vla_instruction']
            except Exception as exc:
                append_jsonl(self.directory/'planner.jsonl', {'step':step,'repair_round':repair+1,'trigger':trigger,
                    'error':type(exc).__name__, 'latency_s':time.monotonic()-planner_started})
                reason = 'planner_error'
                break
        if not pool:
            self.failure_reason = 'no_valid_candidate'
            append_jsonl(self.directory/'selections.jsonl', {'step':step,'error':self.failure_reason})
            raise AgenticInfrastructureError('No candidate has complete valid predictions')
        # max() preserves the earliest candidate on ties. No probabilities enter ranking.
        return self._select(max(pool, key=lambda c:c['satisfied_constraints']), True, reason, started)

    def _select(self, candidate, fallback, reason, started):
        step = candidate['start_step']
        if candidate['satisfied_constraints'] < len(self.questions):
            self._clear_instruction(step, 'selected_unsafe_fallback')
        elif candidate['repair_round'] > 0:
            self.active_instruction = {'instruction': candidate['policy_instruction'],
                'candidate_id': candidate['candidate_id'], 'start_step': step,
                'expires_step': step + 8 * self.config['instruction_ttl_chunks']}
            self.progress_window.reset()
            append_jsonl(self.directory/'recovery.jsonl', {'step': step, 'event': 'instruction_activated',
                         'active_instruction': self.active_instruction})
        self._selected = copy.deepcopy(candidate)
        self.fallbacks += int(fallback)
        append_jsonl(self.directory/'selections.jsonl', {**candidate,'fallback':fallback,
                     'reason':reason,'selection_latency_s':time.monotonic()-started,
                     'stagnation_repair_due':self._stagnation_due,
                     'active_instruction':copy.deepcopy(self.active_instruction)})
        return np.asarray(candidate['commands'],dtype=np.float32)

    def after_step(self, step, monitor, observation):
        super().after_step(step, monitor, observation)
        if not self.ready:
            return
        if self.invalid or self._selected is None:
            self.failure_reason = 'invalid_execution_alignment'
            raise AgenticInfrastructureError('Invalid executed state or monitor alignment')
        offset = step-self._selected['start_step']-1
        if not 0 <= offset < 8:
            raise AgenticInfrastructureError('Executed action outside selected chunk')
        command = self._selected['commands'][offset]
        self._observe(step, observation, command)
        row = {'step':step,'robot_state':self.native_history.rows[-1]['state'],'executed_action':command,
               'policy_instruction':self._selected['policy_instruction'], 'candidate_id':self._selected['candidate_id']}
        self.executed_history.append(row)
        self.executed_history = self.executed_history[-self.config['system1']['history_limit']:]
        append_jsonl(self.directory/'executed-candidates.jsonl', {**row,
                     'oracle_step':step,'maniguard_violated':copy.deepcopy(self.violated)})

    def finish(self, result):
        result['safetyjev_agentic'] = {'decisions':self.decisions,'candidate_attempts':self.candidate_attempts,
            'fallbacks':self.fallbacks,'planner_calls':self.planner_calls,'infrastructure_failure':self.failure_reason}
        if self.failure_reason or self.invalid:
            result.update(status='crashed',success=False)
        super().finish(result)
