"""Synchronous VLA gating, with optional OpenRouter instruction repair.

The guard never consumes ManiGuard oracle state. Only selected candidates enter
the forecast evaluation stream; rejected actions have no observed outcome.
"""
import copy
import hashlib
import math
import time

from .capture import ShadowEpisode
from .io import append_jsonl, write_json


def validate_guard_options(threshold, max_regenerations):
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("guard-threshold must be finite and in [0, 1]")
    if type(max_regenerations) is not int or max_regenerations < 0:
        raise ValueError("max-regenerations must be a nonnegative integer")


def verdict(prediction, forecast_id, threshold):
    """Map a matched, valid score to a binary decision; failures stay unknown."""
    if not isinstance(prediction, dict):
        return "unknown"
    if prediction.get("forecast_id") != forecast_id or prediction.get("error"):
        return "unknown"
    score = prediction.get("score")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1):
        return "unknown"
    return "violated" if score >= threshold else "no_violation"


class GuardedEpisode(ShadowEpisode):
    execution_mode = "guard_regenerate"
    prediction_mode = "online_synchronous_guard"

    def __init__(self, scene, cfg, monitor, observation, episode_seed):
        # capture.OPTIONS is rebound by launch; read it at construction time.
        from . import capture
        options = capture.OPTIONS
        if not options.get("endpoint"):
            raise ValueError("guard_regenerate requires an online predictor")
        if options["recheck_every"] != 0:
            raise ValueError("guard_regenerate v1 checks chunk starts only")
        self.threshold = options["guard_threshold"]
        self.max_regenerations = options["max_regenerations"]
        validate_guard_options(self.threshold, self.max_regenerations)
        super().__init__(scene, cfg, monitor, observation, episode_seed)
        self.attempts = self.rejections = self.acceptances = self.regenerations = 0
        self.unknown_candidates = self.duplicate_candidates = 0
        self.guard_latency_s = self.selection_latency_s = 0.0
        self.termination_reason = None
        self.planner = None
        self.planner_calls = self.planner_failures = 0
        self.planner_latency_s = 0.0
        self.executed_history, self.candidate_history = [], []
        self._selected = None
        if options.get("planner_config"):
            from .planner import OpenRouterPlanner
            self.planner = OpenRouterPlanner(options["planner_config"])
            self.provenance["planner"] = self.planner.metadata()
        self.provenance["guard"] = {
            "threshold": self.threshold, "max_regenerations": self.max_regenerations,
            "acceptance": "all individual constraints must have score < threshold",
            "unknown": "reject and regenerate within the same retry budget",
            "exhaustion": "end simulated episode unsuccessfully without executing the rejected chunk",
            "regeneration": "same observation, task, and episode seed; policy sampling RNG advances",
            "target": "new violation within executed chunk horizon; not detection of past violations",
        }
        if self.planner:
            self.provenance["guard"]["regeneration"] = (
                "OpenRouter repairs the next VLA instruction after each rejection; "
                "same observation and episode seed; unchanged safety constraints")
        write_json(self.directory / "episode.json", self.provenance)

    def before_action(self, *args):
        # select_chunk already checked and recorded this entire execution window.
        pass

    def select_chunk(self, step, observation, initial_chunk, regenerate, action_space, horizon):
        """Return an approved raw chunk, or None; never execute environment steps.

        `regenerate` takes a copy of the observation, with a revised instruction
        when a planner is enabled, and returns another raw VLA chunk. Execution
        applies the same binarization/clipping as make_forecasts.
        """
        import numpy as np

        if horizon <= 0:
            raise ValueError("Execution horizon must be positive")
        started = time.monotonic()
        chunk = initial_chunk
        candidate_observation = copy.deepcopy(observation)
        planner = getattr(self, "planner", None)
        seen = set()
        for attempt in range(self.max_regenerations + 1):
            candidate_id = f"{self.episode_id}:{step}:candidate-{attempt}"
            policy_latency_s = 0.0
            if attempt:
                if planner:
                    if self.planner_calls >= planner.max_calls:
                        self.termination_reason = "planner_budget_exhausted"
                        self.selection_latency_s += time.monotonic() - started
                        append_jsonl(self.directory / "planner.jsonl", {
                            "candidate_id": candidate_id, "error": self.termination_reason})
                        return None
                    self.planner_calls += 1
                    plan_started = time.monotonic()
                    try:
                        plan = planner.plan(repair_context, self.directory)
                    except Exception as exc:
                        from .planner import PlannerError
                        self.planner_failures += 1
                        self.termination_reason = "planner_error"
                        elapsed = time.monotonic() - plan_started
                        self.planner_latency_s += elapsed
                        self.selection_latency_s += time.monotonic() - started
                        append_jsonl(self.directory / "planner.jsonl", {
                            "candidate_id": candidate_id, "latency_s": elapsed,
                            "diagnostics": exc.diagnostics if isinstance(exc, PlannerError) else {},
                            "error": str(exc) if isinstance(exc, PlannerError) else type(exc).__name__})
                        return None
                    self.planner_latency_s += time.monotonic() - plan_started
                    append_jsonl(self.directory / "planner.jsonl", {"candidate_id": candidate_id, **plan})
                    candidate_observation["original_task_instruction"] = observation["task_descriptions"]
                    candidate_observation["task_descriptions"] = plan["vla_instruction"]
                self.regenerations += 1
                before = time.monotonic()
                try:
                    chunk = regenerate(copy.deepcopy(candidate_observation))
                except Exception as exc:
                    self.termination_reason = "policy_error"
                    append_jsonl(self.directory / "decisions.jsonl", {
                        "candidate_id": candidate_id, "start_step": step,
                        "decision": "policy_error", "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise  # upstream records infrastructure failure, with partial logs
                policy_latency_s = time.monotonic() - before
            self.attempts += 1
            # Copy so a policy-owned buffer cannot change after it was checked.
            forecasts, predictions, decisions = [], [], {}
            candidate_hash, error = None, None
            guard_started = time.monotonic()
            try:
                chunk = np.array(chunk, dtype=np.float32, copy=True)
                if chunk.ndim != 2 or chunk.shape[1] != 8 or not len(chunk):
                    raise ValueError("Expected nonempty (H, 8) chunk")
                chunk = chunk[:horizon].copy()
                forecasts = self.make_forecasts(step, candidate_observation, chunk, 0, len(chunk),
                                                action_space, candidate_id, include_global=False)
                if not forecasts:
                    raise ValueError("Cannot accept a candidate with no constraints")
                commands = np.asarray(forecasts[0]["input"]["remaining_actions"], dtype=np.float32)
                candidate_hash = hashlib.sha256(commands.tobytes()).hexdigest()
            except (ValueError, TypeError, OverflowError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            for forecast in forecasts:
                # Persist before calling a potentially failing model service.
                append_jsonl(self.directory / "candidate_forecasts.jsonl", forecast)
                try:
                    prediction = self.predictor.score(forecast)
                except Exception as exc:
                    prediction = {"forecast_id": forecast["forecast_id"], "score": None,
                                  "error": f"{type(exc).__name__}: {exc}"}
                decision = verdict(prediction, forecast["forecast_id"], self.threshold)
                if decision == "unknown":
                    # Ensure malformed/NaN replies cannot poison JSON logs or reports.
                    prediction = {"forecast_id": forecast["forecast_id"], "score": None,
                                  "error": "Invalid, failed, or mismatched guard response"}
                predictions.append(prediction)
                decisions[forecast["constraint_id"]] = decision
                append_jsonl(self.directory / "candidate_predictions.jsonl", prediction)
            elapsed = time.monotonic() - guard_started
            self.guard_latency_s += elapsed
            unknown = error is not None or "unknown" in decisions.values()
            accepted = bool(decisions) and not unknown and all(
                value == "no_violation" for value in decisions.values())
            duplicate = candidate_hash is not None and candidate_hash in seen
            if candidate_hash is not None:
                seen.add(candidate_hash)
            self.duplicate_candidates += int(duplicate)
            self.unknown_candidates += int(unknown)
            self.acceptances += int(accepted)
            self.rejections += int(not accepted)
            append_jsonl(self.directory / "decisions.jsonl", {
                "candidate_id": candidate_id, "start_step": step, "attempt": attempt,
                "decision": "accept" if accepted else "reject",
                "constraint_verdicts": decisions, "error": error,
                "candidate_sha256": candidate_hash, "duplicate_at_this_step": duplicate,
                "guard_latency_s": elapsed, "regeneration_latency_s": policy_latency_s,
                "policy_instruction": candidate_observation["task_descriptions"],
            })
            if planner:
                feedback = [{"constraint_id": f["constraint_id"], "score": p.get("score"),
                             "verdict": decisions[f["constraint_id"]]}
                            for f, p in zip(forecasts, predictions)]
                self.candidate_history.append({
                    "candidate_id": candidate_id, "start_step": step,
                    "policy_instruction": candidate_observation["task_descriptions"],
                    "decision": "accept" if accepted else "reject", "guard_feedback": feedback})
                self.candidate_history = self.candidate_history[-planner.history_limit:]
                repair_context = {
                    "original_task": observation["task_descriptions"],
                    "current_instruction": candidate_observation["task_descriptions"],
                    "start_step": step, "robot_state": observation["states"].tolist(),
                    "remaining_actions": forecasts[0]["input"]["remaining_actions"] if forecasts else [],
                    "action_convention": "absolute joint radians(7) + binarized gripper(1), clipped to controller bounds",
                    "action_frequency_hz": self.cfg.action_frequency,
                    "constraints": copy.deepcopy(self.constraints), "guard_feedback": feedback,
                    "executed_history": copy.deepcopy(self.executed_history),
                    "candidate_history": copy.deepcopy(self.candidate_history),
                    "images": forecasts[0]["input"]["images"] if forecasts else {},
                }
            if accepted:
                # Only the selected trajectory can be joined to subsequent oracle
                # samples. Never label rejected alternatives using this trajectory.
                for forecast, prediction in zip(forecasts, predictions):
                    append_jsonl(self.directory / "forecasts.jsonl", forecast)
                    append_jsonl(self.directory / "predictions.jsonl", prediction)
                if planner:
                    self._selected = {"start_step": step, "commands": forecasts[0]["input"]["remaining_actions"],
                                      "policy_instruction": candidate_observation["task_descriptions"]}
                self.selection_latency_s += time.monotonic() - started
                return chunk
        self.termination_reason = "guard_retries_exhausted"
        self.selection_latency_s += time.monotonic() - started
        return None

    def after_step(self, step, monitor, observation):
        super().after_step(step, monitor, observation)
        selected = getattr(self, "_selected", None)
        if getattr(self, "planner", None) and selected:
            index = step - selected["start_step"] - 1
            if 0 <= index < len(selected["commands"]):
                self.executed_history.append({"step": step, "robot_state": observation["states"].tolist(),
                                              "executed_action": selected["commands"][index],
                                              "policy_instruction": selected["policy_instruction"]})
                self.executed_history = self.executed_history[-self.planner.history_limit:]

    def finish(self, result):
        result["safetyjev_guard"] = {
            "mode": self.execution_mode, "termination_reason": self.termination_reason,
            "candidate_attempts": self.attempts, "accepted_chunks": self.acceptances,
            "rejected_chunks": self.rejections, "regenerations": self.regenerations,
            "unknown_candidates": self.unknown_candidates,
            "duplicate_candidates": self.duplicate_candidates,
            "guard_latency_s": self.guard_latency_s,
            "selection_latency_s": self.selection_latency_s,
            "planner_calls": getattr(self, "planner_calls", 0),
            "planner_failures": getattr(self, "planner_failures", 0),
            "planner_latency_s": getattr(self, "planner_latency_s", 0.0),
        }
        super().finish(result)
