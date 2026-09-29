"""Sidecar observer for the pinned ManiGuard runner; never returns actions."""
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from .io import append_jsonl, read_jsonl, write_json
from .labels import label_forecasts
from .predictors import OpenJevHTTP

OPTIONS = {}


class ShadowEpisode:
    execution_mode = "shadow_no_intervention"
    prediction_mode = "online_synchronous_shadow"

    def __init__(self, scene, cfg, monitor, observation, episode_seed):
        import imageio.v2 as imageio
        from maniguard.utils.ltl_utils import LTLMonitor

        if cfg.state_mode != "joint" or cfg.ik_eef_to_joint or cfg.action_dim != 8:
            raise ValueError("v0.1 supports ManiGuard 8-D absolute joint actions only")
        if cfg.action_frequency <= 0:
            raise ValueError("Invalid action frequency")
        if monitor is None or monitor._monitor is None:
            raise ValueError("Shadow evaluation requires an active ManiGuard monitor")
        self.spec = copy.deepcopy(scene.get("ltl_safety") or {})
        self.constraints = self.spec.get("constraints", [])
        ids = [constraint["id"] for constraint in self.constraints]
        if not ids or len(set(ids)) != len(ids) or "__all__" in ids:
            raise ValueError("Task constraints need unique IDs; __all__ is reserved")
        self.machines = {c["id"]: LTLMonitor(c["ltl"]) for c in self.constraints}
        for constraint in self.constraints:
            constraint["propositions"] = {
                name: self.spec["propositions"][name]
                for name in self.machines[constraint["id"]].ap_list
            }
        for machine in self.machines.values():
            machine.reset()
        self.violated = {cid: False for cid in ids}
        self.invalid = False
        self.last_step = -1
        self.cfg = cfg
        self.imageio = imageio
        self.history = []
        self.last_state = observation["states"].tolist()
        self.last_observation_step = 0
        self.episode_id = uuid.uuid4().hex
        self.directory = Path(OPTIONS["output"]).resolve() / self.episode_id
        (self.directory / "frames").mkdir(parents=True, exist_ok=False)
        self.provenance = copy.deepcopy(OPTIONS["provenance"])
        self.provenance.update({
            "episode_id": self.episode_id, "scene_name": scene["name"],
            "pipeline": scene.get("pipeline"), "episode_seed": episode_seed,
            "scene_file_sha256": hashlib.sha256(Path(scene["scene_file"]).read_bytes()).hexdigest(),
            "config": vars(cfg), "specification": self.spec,
            "mode": self.execution_mode, "recheck_every": OPTIONS["recheck_every"],
            "clock": "synchronous simulation; wall time advances during model calls, simulation does not",
            "label_semantics": "raw bad-prefix rejection; not engagement-gated episode safety",
        })
        write_json(self.directory / "episode.json", self.provenance)
        self.predictor = None
        if OPTIONS.get("endpoint"):
            self.predictor = OpenJevHTTP(OPTIONS["endpoint"], OPTIONS["model_id"],
                                       OPTIONS["input_mode"], OPTIONS["timeout"])
            write_json(self.directory / "prediction.meta.json", {
                "mode": self.prediction_mode, "input_mode": OPTIONS["input_mode"],
                "model_id": OPTIONS["model_id"], "model_revision": OPTIONS["predictor_revision"],
                "endpoint": OPTIONS["endpoint"],
            })
        self.after_step(0, monitor, observation)
        if self.invalid:
            raise ValueError("Initial monitor labels unavailable; see oracle.jsonl")

    def before_action(self, step, observation, chunk, ci, chunk_len, action_space):
        cadence = OPTIONS["recheck_every"]
        if ci != 0 and (cadence == 0 or ci % cadence):
            return
        for forecast in self.make_forecasts(step, observation, chunk, ci, chunk_len, action_space):
            # Persist the exact pre-action input, even when inference fails.
            append_jsonl(self.directory / "forecasts.jsonl", forecast)
            if self.predictor:
                append_jsonl(self.directory / "predictions.jsonl", self.predictor.score(forecast))

    def make_forecasts(self, step, observation, chunk, ci, chunk_len, action_space,
                       candidate_id=None, include_global=True):
        import numpy as np

        # Exactly the commands the joint-only runner will execute, using a COPY.
        actions = np.array(chunk[ci:chunk_len], dtype=np.float32, copy=True)
        if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions):
            raise ValueError("Expected nonempty (H, 8) chunk")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Nonfinite proposed action")
        if self.cfg.gripper_binarize:
            g = actions[:, -1]
            actions[:, -1] = np.where(np.abs(g) > .01, np.sign(g), -1.0)
        actions = np.clip(actions, action_space.low, action_space.high)
        if not np.all(np.isfinite(actions)):
            raise ValueError("Nonfinite proposed action")
        images = {}
        for key in ("overview_image", "wrist_images"):
            relative = f"frames/{step:07d}-{key}.png"
            self.imageio.imwrite(self.directory / relative, observation[key])
            images[key] = relative
        constraints = self.constraints + ([{
            "id": "__all__", "ltl": self.spec.get("combined_ltl", ""),
            "description": "Any of the task safety constraints",
            "clauses": self.constraints,
        }] if include_global else [])
        forecasts = []
        window_id = candidate_id or f"{self.episode_id}:{step}"
        for constraint in constraints:
            forecast = {
                "schema_version": 1,
                "forecast_id": f"{window_id}:{constraint['id']}",
                "episode_id": self.episode_id, "constraint_id": constraint["id"],
                "start_step": step, "end_step": step + len(actions),
                "captured_monotonic_s": time.monotonic(),
                "input": {
                    "robot_state": observation["states"].tolist(),
                    "past_robot_states": copy.deepcopy(self.history[-4:]),
                    "task_instruction": observation["task_descriptions"],
                    "images": images, "remaining_actions": actions.tolist(),
                    "action_frequency_hz": self.cfg.action_frequency,
                    "action_convention": "absolute joint radians(7) + binarized gripper(1), clipped to controller bounds",
                    "constraint": constraint,
                },
            }
            forecasts.append(forecast)
        return forecasts

    def after_step(self, step, monitor, observation):
        error = None
        entry = monitor._ltl_log[-1] if monitor is not None and monitor._ltl_log else None
        try:
            if self.invalid or step != self.last_step + 1:
                raise ValueError("Monitor history has a gap")
            if monitor._monitor is None or not entry or entry["step"] != step:
                raise ValueError("Missing current ManiGuard monitor sample")
            ap = entry["ap"]
            for cid, machine in self.machines.items():
                if not set(machine.ap_list).issubset(ap):
                    raise ValueError("Required proposition missing from ground truth")
                result = machine.step(ap)
                self.violated[cid] |= bool(result["doomed"])
            if any(self.violated.values()) != bool(monitor.violated):
                raise ValueError("Per-constraint monitors disagree with combined task monitor")
        except Exception as exc:
            self.invalid = True
            error = f"{type(exc).__name__}: {exc}"
        self.last_step = step
        append_jsonl(self.directory / "oracle.jsonl", {
            "step": step, "valid": not self.invalid, "error": error,
            "violated": {**self.violated, "__all__": bool(monitor.violated) if monitor else False},
            "ap": entry.get("ap") if entry else None,
            "monitor_state": entry.get("state") if entry else None,
        })
        if step > 0:
            self.history.append({"step": self.last_observation_step, "robot_state": self.last_state})
            self.history = self.history[-4:]
            self.last_state = observation["states"].tolist()
            self.last_observation_step = step

    def finish(self, result):
        write_json(self.directory / "maniguard_result.json", result)
        forecasts = self.directory / "forecasts.jsonl"
        labels = label_forecasts(read_jsonl(forecasts) if forecasts.exists() else [],
                                 read_jsonl(self.directory / "oracle.jsonl"))
        for row in labels:
            append_jsonl(self.directory / "labels.jsonl", row)
        # End marker is required by the reporting CLI: crashed processes cannot
        # silently contribute an apparently complete episode.
        write_json(self.directory / "complete.json", {"final_step": result["steps"],
                   "monitor_valid": not self.invalid, "status": result["status"]})
