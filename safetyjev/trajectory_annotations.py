"""Trajectory-level GT: observed state intervals and temporal first rejections."""
import json
from collections import Counter
from pathlib import Path

from .data_preparation import _episodes, _local_path, _predicate_value, _sha256, _trace
from .io import write_json


def _boolean(record, ap):
    value = record.get("ap", {}).get(ap)
    if type(value) is not bool:
        raise ValueError("Missing or non-Boolean AP: " + ap)
    return value


def _observed_intervals(log, is_active):
    intervals, start = [], None
    for record in log:
        active = is_active(record)
        if active and start is None:
            start = record["step"]
        elif not active and start is not None:
            intervals.append({"start_timestep": start, "end_timestep": record["step"] - 1,
                              "end_reason": "recovered"})
            start = None
    if start is not None:
        intervals.append({"start_timestep": start, "end_timestep": log[-1]["step"],
                          "end_reason": "trajectory_end"})
    return intervals


def state_intervals(log, ap, violating_value):
    if type(violating_value) is not bool:
        raise ValueError("violating_value must be Boolean")
    return _observed_intervals(log, lambda record: _boolean(record, ap) == violating_value)


def derived_intervals(log, checks):
    events = []
    for check in checks:
        if "all_of" not in check:
            raise ValueError("Derived checks require an explicit AP conjunction")
        for interval in _observed_intervals(log, lambda record: _predicate_value(record, check)):
            events.append({"type": check["type"], "label_kind": "state_interval",
                           "origin": "derived_current_check", "all_of": check["all_of"], **interval})
    return events


def until_first_rejection(log, maintain_ap, release_ap):
    """Bad prefix of maintain U release; end-of-trace nonrelease is not a rejection."""
    for record in log:
        if _boolean(record, release_ap):
            return None
        if not _boolean(record, maintain_ap):
            return record["step"]
    return None


def _formula(value):
    # Only unary invariants and atomic until clauses are supported here.
    return "".join(value.split()).replace("(", "").replace(")", "")


def annotate_trace(trace, rules):
    log = trace["log"]
    if not log or [r.get("step") for r in log] != list(range(len(log))):
        raise ValueError("Annotation requires a complete contiguous AP trace")
    clauses = {c["id"]: c["ltl"] for c in trace["constraints"]}
    if {r["constraint_id"] for r in rules} != set(clauses):
        raise ValueError("Annotation rules must cover exactly the recorded safety clauses")
    events = []
    for rule in rules:
        if _formula(clauses[rule["constraint_id"]]) != _formula(rule["ltl"]):
            raise ValueError("Recorded formula differs from annotation rule")
        common = {"type": rule["type"], "constraint_id": rule["constraint_id"],
                  "origin": "historical_specification",
                  "label_kind": rule["kind"]}
        if rule["kind"] == "state_interval":
            expected = "G(" + ("!" if rule["violating_value"] else "") + rule["ap"] + ")"
            if _formula(expected) != _formula(rule["ltl"]):
                raise ValueError("State annotation polarity differs from the recorded invariant")
            common.update(ap=rule["ap"], violating_value=rule["violating_value"])
            events.extend({**common, **interval} for interval in
                          state_intervals(log, rule["ap"], rule["violating_value"]))
        elif rule["kind"] == "temporal_first_rejection":
            expected = rule["maintain_ap"] + "U" + rule["release_ap"]
            if _formula(expected) != _formula(rule["ltl"]):
                raise ValueError("Temporal annotation differs from the recorded until clause")
            step = until_first_rejection(log, rule["maintain_ap"], rule["release_ap"])
            if step is not None:
                events.append({**common, "start_timestep": step,
                               "maintain_ap": rule["maintain_ap"], "release_ap": rule["release_ap"]})
        else:
            raise ValueError("Unsupported annotation kind: " + rule["kind"])
    events.sort(key=lambda e: (e["start_timestep"], e["type"]))
    first = events[0]["start_timestep"] if events else None
    if bool(events) != trace["violated"] or first != trace["violation_step"]:
        raise ValueError("Derived events disagree with the historical monitor verdict")
    return events


def build_trajectory_gt(root, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING").write_text("Trajectory annotation is in progress.\n")
    totals, types, episode_types = Counter(), Counter(), Counter()
    with (output / "trajectories.jsonl").open("w") as stream:
        for episode in _episodes(root):
            _trace(root, episode)
            path = _local_path(root, episode["trace"])
            trace = json.loads(path.read_text())
            digest = _sha256(path)
            expected = next(f["sha256"] for f in episode["files"] if f["path"] == episode["trace"])
            if digest != expected:
                raise ValueError("Raw AP trace changed since the reviewed copy")
            events = annotate_trace(trace, config["violations"])
            # Independently check each first event against the archived attribution.
            attribution = [f["path"] for f in episode["files"] if f["path"].endswith("_ltl_attribution.json")]
            if len(attribution) != 1:
                raise ValueError("Expected per-constraint attribution for this collection")
            for clause in json.loads(_local_path(root, attribution[0]).read_text())["constraints"]:
                starts = [e["start_timestep"] for e in events if e["constraint_id"] == clause["id"]]
                first = min(starts) if starts else None
                if first != clause.get("first_violation_step") or bool(starts) != clause["violated"]:
                    raise ValueError("Per-constraint attribution mismatch: " + episode["id"])
            extra = derived_intervals(trace["log"], config.get("derived_checks", []))
            totals["historical_events"] += len(events)
            totals["derived_current_events"] += len(extra)
            totals["derived_positive_episodes"] += bool(extra)
            totals["derived_positive_timesteps"] += sum(e["end_timestep"] - e["start_timestep"] + 1 for e in extra)
            events.extend(extra)
            events.sort(key=lambda e: (e["start_timestep"], e["type"]))
            observations = {}
            for camera, suffix in (("overview", "_main.mp4"), ("wrist", "_wrist.mp4")):
                matches = [f["path"] for f in episode["files"] if f["path"].endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError("Expected exactly one " + camera + " video")
                observations[camera] = matches[0]
            task = Path(episode["trace"]).parent.name
            row = {"schema_version": 1, "trajectory_id": episode["id"],
                   "safety": "unsafe" if events else "safe", "last_timestep": episode["steps"],
                   "observations": observations, "video_fps": episode["video_fps"],
                   "control_hz": episode.get("control_hz"), "ap_trace": episode["trace"],
                   "ap_trace_sha256": digest, "violations": events,
                   "metadata": {"family": config["family"], "base_task": task,
                                "split_group": config["family"] + "/" + task,
                                "historical_monitor_violated": trace["violated"],
                                "safety_scope": "historical_clauses_and_configured_derived_checks",
                                "source": episode.get("source"), "condition": episode.get("condition"),
                                "model": episode.get("model"), "regime": episode.get("regime")}}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            totals["trajectories"] += 1
            totals[row["safety"]] += 1
            totals["events"] += len(events)
            types.update(e["type"] for e in events)
            episode_types.update({e["type"] for e in events})
    write_json(output / "annotation_types.json", config)
    summary = {**totals, "events_by_type": dict(types), "episodes_by_type": dict(episode_types),
               "raw_root": str(Path(root).resolve()),
               "raw_manifest_sha256": _sha256(Path(root) / "manifest.jsonl"),
               "interval_endpoints": "inclusive; trajectory_end does not imply recovery",
               "temporal_events": "first rejection only; no invented recovery interval",
               "gt_basis": "historical clauses checked against attribution; additional current-state checks derived from the same AP trace"}
    write_json(output / "dataset_metadata.json", summary)
    (output / "BUILDING").unlink()
    return summary
