"""Live OpenRouter probe on saved shadow inputs; never a live robot evaluation."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from safetyjev.guard import validate_guard_options, verdict
from safetyjev.io import read_jsonl, write_json
from safetyjev.planner import OpenRouterPlanner, PlannerError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_guard_options(args.threshold, 0)
    if args.output.exists():
        raise ValueError("Choose a new output file")
    meta = json.loads((args.episode / "episode.json").read_text())
    if meta["mode"] != "shadow_no_intervention":
        raise ValueError("Probe expects recorded shadow trajectories")
    rows = [r for r in read_jsonl(args.episode / "forecasts.jsonl")
            if r["start_step"] <= args.step and r["constraint_id"] != "__all__"]
    current = [r for r in rows if r["start_step"] == args.step]
    if not current or len({r["constraint_id"] for r in current}) != len(current):
        raise ValueError("Need exactly one saved forecast per constraint at this step")
    predictions = {r["forecast_id"]: r for r in read_jsonl(args.episode / "predictions.jsonl")}
    feedback = [{"constraint_id": r["constraint_id"], "score": predictions[r["forecast_id"]]["score"],
                 "verdict": verdict(predictions[r["forecast_id"]], r["forecast_id"], args.threshold)}
                for r in current]
    if not any(r["verdict"] == "violated" for r in feedback):
        raise ValueError("This selected threshold does not reject the saved candidate")
    planner = OpenRouterPlanner(json.loads(args.config.read_text()))
    data = current[0]["input"]
    states, actions = {}, {}
    for row in rows:
        start, inputs = row["start_step"], row["input"]
        states[start] = inputs["robot_state"]
        for previous in inputs["past_robot_states"]:
            states[previous["step"]] = previous["robot_state"]
        for index, action in enumerate(inputs["remaining_actions"]):
            end = start + index + 1
            if end <= args.step:
                actions[end] = (action, inputs["task_instruction"])
    history = [{"step": step, "robot_state": states[step], "executed_action": actions[step][0],
                "policy_instruction": actions[step][1]}
               for step in sorted(states.keys() & actions.keys())[-planner.history_limit:]]
    context = {"original_task": data["task_instruction"], "current_instruction": data["task_instruction"],
               "start_step": args.step, "robot_state": data["robot_state"],
               "remaining_actions": data["remaining_actions"], "action_convention": data["action_convention"],
               "action_frequency_hz": data["action_frequency_hz"],
               "constraints": [r["input"]["constraint"] for r in current], "guard_feedback": feedback,
               "executed_history": history, "candidate_history": [], "images": data["images"]}
    report = {"mode": "live_planner_on_recorded_shadow_inputs", "live_simulation": False,
              "source_episode": meta["episode_id"], "step": args.step, "threshold": args.threshold,
              "threshold_note": "Deliberate rejection-path diagnostic; not a tuned safety threshold",
              "planner": planner.metadata(), "guard_feedback": feedback,
              "source_sha256": {name: hashlib.sha256((args.episode / name).read_bytes()).hexdigest()
                                for name in ("forecasts.jsonl", "predictions.jsonl")}}
    started = time.monotonic()
    try:
        report["plan"] = planner.plan(context, args.episode)
        report["status"] = "passed"
    except PlannerError as exc:
        report.update(status="failed", error=str(exc), diagnostics=exc.diagnostics)
    report["elapsed_s"] = time.monotonic() - started
    write_json(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "error", "elapsed_s")}))


if __name__ == "__main__":
    main()
