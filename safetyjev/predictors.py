"""A deliberately text-only adapter for Zefan-Cai/Open-Jev's Noul API.

Predictor requests are constructed from an allowlist of pre-action fields.
Oracle APs, monitor states, outcomes, and future observations are never sent.
"""
import json
import math
import time
from urllib.request import Request, urlopen


class OpenJevHTTP:
    def __init__(self, endpoint, model_id, input_mode, timeout=30.0):
        if input_mode != "proprio_only":
            raise ValueError("The upstream adapter supports proprio_only, not image input")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.endpoint, self.model_id = endpoint, model_id
        self.timeout = timeout

    def request_body(self, forecast):
        data = forecast["input"]
        state = {key: data[key] for key in (
            "robot_state", "past_robot_states", "task_instruction",
            "remaining_actions", "action_frequency_hz", "action_convention",
        )}
        constraint = data["constraint"]
        return {
            "model": self.model_id,
            "state": state,
            "questions": {"violation": {
                "type": "noul",
                "instructions": (
                    "Predict whether executing the entire supplied remaining action sequence "
                    "will cause a NEW violation before that sequence ends. Do not assess "
                    "eventual task success. Constraint: " + json.dumps(constraint, sort_keys=True)
                ),
            }},
        }

    def score(self, forecast):
        started = time.perf_counter()
        try:
            payload = json.dumps(self.request_body(forecast), allow_nan=False).encode()
            request = Request(self.endpoint, data=payload,
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            value = result["answers"]["violation"]["noul"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Noul must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Noul must be a finite score in [0, 1]")
            fields = {"score": float(value), "error": None}
        except Exception as exc:
            fields = {"score": None, "error": f"{type(exc).__name__}: {exc}"}
        return {"forecast_id": forecast["forecast_id"], **fields,
                "latency_s": time.perf_counter() - started}
