"""Package reviewed families into grouped Open-Jev Noul training splits."""
from contextlib import ExitStack
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from .data_preparation import _sha256, iter_ap_windows
from .io import write_json
from .openjev_data import to_openjev_record


def assign_group_splits(groups, fractions, seed):
    allowed = {"train", "calibration", "validation", "test", "ood"}
    if not fractions or set(fractions) - allowed or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must be finite, valid, and sum to one")
    families = defaultdict(list)
    for group in sorted(set(groups)):
        families[group.split("/", 1)[0]].append(group)
    assignments = {}
    names = list(fractions)
    for family_groups in families.values():
        family_groups.sort(key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        exact = [len(family_groups) * fractions[k] for k in names]
        counts = [math.floor(n) for n in exact]
        remaining = len(family_groups) - sum(counts)
        order = sorted(range(len(names)), key=lambda i: (-(exact[i] - counts[i]), i))
        for i in order[:remaining]:
            counts[i] += 1
        offset = 0
        for name, count in zip(names, counts):
            for group in family_groups[offset:offset + count]:
                assignments[group] = name
            offset += count
    return assignments


def build_mixture(recipe, output):
    sources = recipe["sources"]
    if not sources or len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Provide uniquely named data sources")
    window = recipe["window"]
    if set(window) != {"history_frames", "frame_stride", "sample_stride"} or any(
        type(v) is not int or v <= 0 for v in window.values()
    ):
        raise ValueError("Specify all three positive integer window parameters")
    prepared, groups, identities = [], set(), set()
    for source in sources:
        config = json.loads(Path(source["queries"]).read_text())
        gt_path = Path(source["trajectories"])
        with gt_path.open() as stream:
            gt = [json.loads(line) for line in stream if line.strip()]
        if not gt or any(r["metadata"]["family"] != config["family"] for r in gt):
            raise ValueError("Source GT and query families must agree")
        groups.update(r["metadata"]["split_group"] for r in gt)
        identity = (_sha256(Path(source["raw"]) / "manifest.jsonl"), _sha256(gt_path), _sha256(source["queries"]))
        if identity in identities:
            raise ValueError("The same raw/GT/query source was included twice")
        identities.add(identity)
        if not isinstance(source.get("license"), str) or not source["license"].strip():
            raise ValueError("Specify source license metadata; use unspecified if it is not established")
        prepared.append((source, config, identity))
    splits = assign_group_splits(groups, recipe["splits"], recipe["seed"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING").write_text("Multi-family package build is in progress.\n")
    counts, by_query, by_source = Counter(), defaultdict(Counter), Counter()
    resources = {}
    with ExitStack() as stack:
        streams = {name: stack.enter_context((output / (name + ".jsonl")).open("w")) for name in recipe["splits"]}
        for source, config, identity in prepared:
            name = source["name"]
            raw = Path(source["raw"]).resolve()
            resources[name] = {"raw_root": os.path.relpath(raw, output), "family": config["family"],
                               "raw_manifest_sha256": identity[0], "trajectory_gt_sha256": identity[1],
                               "queries_sha256": identity[2]}
            provenance = {"input_sha256": identity[1], "source_url": source.get("source_url", raw.as_uri()),
                          "license": source["license"],
                          "split_policy": f"family_base_task_hash_rank_v1_seed_{recipe['seed']}"}
            for sample in iter_ap_windows(raw, config, trajectory_index=source["trajectories"], **window):
                split = splits[sample["metadata"]["split_group"]]
                record = to_openjev_record(sample, split=split, source=name, provenance=provenance)
                record["id"] = name + ":" + record["id"]
                record["state"]["observation_window"]["resource_id"] = name
                streams[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                answer = sample["target"]["answer"]
                counts[split] += 1
                by_source[name] += 1
                by_query[f"{split}/{config['family']}/{sample['metadata']['query_id']}"][str(answer)] += 1
    if not sum(counts.values()):
        raise ValueError("The recipe produced no complete windows")
    write_json(output / "group_splits.json", splits)
    write_json(output / "recipe.json", recipe)
    summary = {"schema_version": 1, "samples": sum(counts.values()), "splits": dict(counts),
               "groups_per_split": dict(Counter(splits.values())), "by_source": dict(by_source),
               "by_query": dict(by_query), "resources": resources, "window": window,
               "split_files": {name: name + ".jsonl" for name in streams},
               "file_sha256": {name: _sha256(output / (name + ".jsonl")) for name in streams},
               "model_interface": "Noul record plus decoded RGB observations; no source metadata in model payload"}
    write_json(output / "dataset_metadata.json", summary)
    (output / "BUILDING").unlink()
    return summary
