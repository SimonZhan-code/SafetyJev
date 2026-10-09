"""Build current-AP classification indices from archived observation traces."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .io import write_json


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Raw resource is outside the dataset: " + str(relative))
    return path


def _episodes(root):
    seen = set()
    with (Path(root) / "manifest.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            episode = json.loads(line)
            if episode["id"] in seen:
                raise ValueError("Duplicate episode ID: " + episode["id"])
            seen.add(episode["id"])
            yield episode
    if not seen:
        raise ValueError("Raw manifest has no episodes")


def _trace(root, episode):
    data = json.loads(_local_path(root, episode["trace"]).read_text())
    log = data["log"]
    if len(log) != episode["steps"] + 1:
        raise ValueError("Trace does not cover the complete recorded episode")
    for step, record in enumerate(log):
        if type(record.get("step")) is not int or record["step"] != step:
            raise ValueError("Trace steps must be contiguous from zero")
        ap = record.get("ap")
        if not isinstance(ap, dict) or not ap or any(type(v) is not bool for v in ap.values()):
            raise ValueError("Missing or non-Boolean AP supervision")
    return log


def _predicate_aps(definition):
    if ("ap" in definition) == ("all_of" in definition):
        raise ValueError("Specify exactly one of ap or all_of")
    if "ap" in definition:
        ap = definition["ap"]
        if not isinstance(ap, str) or not ap.strip():
            raise ValueError("AP name must be nonempty text")
        return {ap}
    terms = definition["all_of"]
    if not isinstance(terms, dict) or not terms or any(
        not isinstance(k, str) or not k.strip() or type(v) is not bool for k, v in terms.items()
    ):
        raise ValueError("all_of requires nonempty AP names mapped to Boolean values")
    return set(terms)


def _predicate_value(record, definition):
    names = _predicate_aps(definition)
    values = {name: record.get("ap", {}).get(name) for name in names}
    if any(type(value) is not bool for value in values.values()):
        raise ValueError("Missing or non-Boolean AP in current-state predicate")
    if "ap" in definition:
        return values[definition["ap"]]
    return all(values[name] == expected for name, expected in definition["all_of"].items())


def _queries(config):
    if config.get("schema_version") != 1 or not config.get("family"):
        raise ValueError("Query configuration requires schema_version=1 and family")
    queries = config.get("queries", [])
    if not queries:
        raise ValueError("At least one AP question is required")
    seen = set()
    for query in queries:
        for key in ("id", "question"):
            if not isinstance(query.get(key), str) or not query[key].strip():
                raise ValueError("Missing query " + key)
        _predicate_aps(query)
        if "question_by_scene" in query:
            mapping = query["question_by_scene"]
            if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                for k, v in mapping.items()
            ):
                raise ValueError("question_by_scene requires nonempty scene keys and question strings")
        if type(query.get("yes_if")) is not bool:
            raise ValueError("yes_if must explicitly specify a Boolean AP value")
        if query["id"] in seen:
            raise ValueError("Duplicate query ID: " + query["id"])
        seen.add(query["id"])
    return queries


def validate_raw(root, verify_hashes=False):
    """Check archive integrity; this does not establish physical label correctness."""
    totals = Counter()
    values = defaultdict(Counter)
    missing_config = []
    for episode in _episodes(root):
        listed = {item["path"] for item in episode["files"]}
        if episode["trace"] not in listed:
            raise ValueError("Trace is not listed in the file manifest")
        for item in episode["files"]:
            path = _local_path(root, item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError("Missing or changed raw file: " + item["path"])
            if verify_hashes and _sha256(path) != item["sha256"]:
                raise ValueError("Checksum mismatch: " + item["path"])
            totals["files"] += 1
            totals["bytes"] += item["bytes"]
        for record in _trace(root, episode):
            totals["timestep_records"] += 1
            for name, value in record["ap"].items():
                values[name]["true" if value else "false"] += 1
        totals["episodes"] += 1
        if episode.get("missing_eval_config"):
            missing_config.append(episode["id"])
    return {**totals, "ap_counts": dict(values), "missing_eval_config": missing_config,
            "hashes_verified": verify_hashes}


def _window_sources(root, trajectory_index, family):
    if trajectory_index is None:
        yield from _episodes(root)
        return
    seen = set()
    with Path(trajectory_index).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            gt = json.loads(line)
            if gt["trajectory_id"] in seen:
                raise ValueError("Duplicate trajectory GT ID")
            seen.add(gt["trajectory_id"])
            if gt["metadata"]["family"] != family:
                raise ValueError("Trajectory family does not match the AP question configuration")
            if _sha256(_local_path(root, gt["ap_trace"])) != gt["ap_trace_sha256"]:
                raise ValueError("AP trace no longer matches the trajectory GT")
            yield {**gt["metadata"], "id": gt["trajectory_id"], "trace": gt["ap_trace"],
                   "steps": gt["last_timestep"], "video_fps": gt["video_fps"],
                   "control_hz": gt.get("control_hz"), "trajectory_gt_id": gt["trajectory_id"],
                   "files": [{"path": path} for path in gt["observations"].values()]}
    if not seen:
        raise ValueError("Trajectory GT index is empty")


def iter_ap_windows(root, config, *, history_frames, frame_stride, sample_stride,
                    limit_episodes=None, trajectory_index=None):
    """Yield causal observation windows with endpoint AP answers, one query per row."""
    queries = _queries(config)
    for name, value in (("history_frames", history_frames), ("frame_stride", frame_stride),
                        ("sample_stride", sample_stride)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if limit_episodes is not None and (type(limit_episodes) is not int or limit_episodes <= 0):
        raise ValueError("limit_episodes must be a positive integer")
    first = (history_frames - 1) * frame_stride
    family = config["family"]
    query_signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    for index, episode in enumerate(_window_sources(root, trajectory_index, family)):
        if limit_episodes is not None and index >= limit_episodes:
            break
        log = _trace(root, episode)
        required = set().union(*(_predicate_aps(q) for q in queries))
        if any(not required.issubset(record["ap"]) for record in log):
            raise ValueError("An AP required by the questions is absent: " + episode["id"])
        videos = {}
        for camera, suffix in (("overview", "_main.mp4"), ("wrist", "_wrist.mp4")):
            matches = [f["path"] for f in episode["files"] if f["path"].endswith(suffix)]
            if len(matches) != 1 or not _local_path(root, matches[0]).is_file():
                raise ValueError("Expected exactly one " + camera + " video")
            videos[camera] = matches[0]
        # The archived scene folder is task_NNNN/<condition>_ltl.json.
        task = Path(episode["trace"]).parent.name
        scene = task + "/" + Path(episode["trace"]).name.removesuffix("_ltl.json")
        questions = {}
        for query in queries:
            if "question_by_scene" in query:
                if scene not in query["question_by_scene"]:
                    raise ValueError("Missing scene-specific question: " + scene + "/" + query["id"])
                questions[query["id"]] = query["question_by_scene"][scene]
            else:
                questions[query["id"]] = query["question"]
        for step in range(first, len(log), sample_stride):
            frame_indices = list(range(step - first, step + 1, frame_stride))
            for query in queries:
                value = _predicate_value(log[step], query)
                yield {
                    "schema_version": 1,
                    "sample_id": f"{family}:{query_signature}:{episode['id']}:{step}:{query['id']}",
                    "input": {"question": questions[query["id"]]},
                    "media": {"videos": videos.copy(), "frame_indices": frame_indices,
                              "video_fps": episode["video_fps"], "control_hz": episode.get("control_hz")},
                    "target": {"answer": int(value == query["yes_if"]), "semantics": "question_yes_no"},
                    "metadata": {"episode_id": episode["id"], "family": family,
                                 "base_task": task, "split_group": family + "/" + task,
                                 "condition": episode.get("condition"), "model": episode.get("model"),
                                 "regime": episode.get("regime"), "source": episode.get("source"),
                                 "query_id": query["id"], "ap": query.get("ap"),
                                 "ap_value": value if "ap" in query else None,
                                 "predicate_value": value,
                                 "component_ap_values": {name: log[step]["ap"][name]
                                                         for name in sorted(_predicate_aps(query))},
                                 "query_step": step, "yes_if": query["yes_if"],
                                 "trajectory_gt_id": episode.get("trajectory_gt_id")},
                }


def model_payload(sample, decoded_observations):
    """The training adapter decodes media first; only this payload enters the model."""
    cameras = set(sample["media"].get("image_refs", sample["media"].get("videos", {})))
    frames = len(sample["media"]["frame_indices"])
    if set(decoded_observations) != cameras or any(len(v) != frames for v in decoded_observations.values()):
        raise ValueError("Decoded observations must match the indexed cameras and frame count")
    return {"question": sample["input"]["question"],
            "observations": {camera: decoded_observations[camera] for camera in sorted(cameras)}}


def export_ap_dataset(root, config, output, **window_options):
    _queries(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING").write_text("Index export is incomplete until dataset_metadata.json exists.\n")
    counts = defaultdict(Counter)
    episodes, groups = set(), set()
    total = 0
    with (output / "samples.jsonl").open("w") as stream:
        for sample in iter_ap_windows(root, config, **window_options):
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[sample["metadata"]["query_id"]][str(sample["target"]["answer"])] += 1
            episodes.add(sample["metadata"]["episode_id"])
            groups.add(sample["metadata"]["split_group"])
            total += 1
    if not total:
        raise ValueError("No complete history windows are available for these parameters")
    write_json(output / "queries.json", config)
    summary = {"samples": total, "episodes": len(episodes), "split_groups": sorted(groups),
               "by_query": dict(counts), "window_options": {k: str(v) if isinstance(v, Path) else v
                                                           for k, v in window_options.items()},
               "raw_manifest_sha256": _sha256(Path(root) / "manifest.jsonl"),
               "raw_root": str(Path(root).resolve()), "split_assignment": None,
               "label_time": "window endpoint", "media": "references to unchanged raw videos",
               "model_input": "question and decoded observations only"}
    if window_options.get("trajectory_index") is not None:
        summary["trajectory_gt_sha256"] = _sha256(window_options["trajectory_index"])
    write_json(output / "dataset_metadata.json", summary)
    (output / "BUILDING").unlink()
    return summary
