import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from .io import append_jsonl, read_jsonl, write_json
from .labels import label_forecasts
from .metrics import evaluate
from .predictors import OpenJevHTTP


def episodes(root):
    root = Path(root)
    paths = [root] if (root / "episode.json").exists() else sorted(p.parent for p in root.glob("*/episode.json"))
    if not paths:
        raise ValueError("No captured episodes found")
    return paths


def model_args(parser):
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--predictor-revision", required=True, help="Checkpoint revision/hash; provenance, not server attestation")
    parser.add_argument("--input-mode", required=True, choices=["proprio_only"])
    parser.add_argument("--timeout", type=float, default=30)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafetyJev shadow prediction evaluation")
    subs = parser.add_subparsers(dest="command", required=True)
    verify = subs.add_parser("verify-integration")
    verify.add_argument("--maniguard-root", required=True)
    capture = subs.add_parser("capture", help="Run original policy with shadow sidecars; use -- before ManiGuard args")
    capture.add_argument("--maniguard-root", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--provenance", required=True, help="JSON with policy repo/revision and benchmark revision")
    capture.add_argument("--recheck-every", type=int, default=0, help="0: chunk starts only; N: every N executed actions")
    capture.add_argument("--online-predictor", help="JSON with endpoint/model_id/predictor_revision/input_mode/timeout")
    capture.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    predict = subs.add_parser("predict", help="Score saved pre-action inputs only, without oracle data")
    predict.add_argument("--episodes", required=True)
    predict.add_argument("--name", required=True, help="New prediction run name")
    predict.add_argument("--eligible-only", action="store_true",
                         help="Replay only evaluable windows from complete episodes; oracle data stays outside the scorer")
    model_args(predict)
    report = subs.add_parser("report")
    report.add_argument("--episodes", required=True)
    report.add_argument("--predictions", default="predictions", help="Prediction filename stem")
    report.add_argument("--threshold", type=float, default=.5)
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-integration":
        from .maniguard import COMMIT, instrument, verify_sources
        verify_sources(args.maniguard_root)
        path = Path(args.maniguard_root) / "maniguard/eval/benchmark.py"
        instrument(path.read_text())
        print(json.dumps({"status": "source_hooks_verified", "maniguard_commit": COMMIT,
                          "gpu_simulator_tested": False}))
    elif args.command == "capture":
        from .maniguard import COMMIT, launch
        if args.recheck_every < 0:
            raise ValueError("recheck-every cannot be negative")
        provenance = json.loads(Path(args.provenance).read_text())
        for field in ("policy_repo", "policy_revision", "benchmark_revision"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip() or value in ("REQUIRED", "main", "latest"):
                raise ValueError(f"Pin {field} in provenance before capturing")
        provenance["maniguard_reference_commit"] = COMMIT
        provenance["source_hashes"] = {
            name: hashlib.sha256((Path(args.maniguard_root) / name).read_bytes()).hexdigest()
            for name in ("maniguard/eval/benchmark.py", "maniguard/utils/ltl_utils.py",
                         "maniguard/utils/safety_monitor.py")
        }
        options = {"output": str(Path(args.output).resolve()), "provenance": provenance,
                   "recheck_every": args.recheck_every}
        if args.online_predictor:
            config = json.loads(Path(args.online_predictor).read_text())
            for key in ("endpoint", "model_id", "predictor_revision", "input_mode"):
                if not config.get(key):
                    raise ValueError("Missing online predictor field " + key)
                options[key] = config[key]
            options["timeout"] = config.get("timeout", 30)
            OpenJevHTTP(options["endpoint"], options["model_id"], options["input_mode"], options["timeout"])
        benchmark_args = args.benchmark_args
        if benchmark_args[:1] == ["--"]:
            benchmark_args = benchmark_args[1:]
        launch(args.maniguard_root, benchmark_args, options)
    elif args.command == "predict":
        if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
            raise ValueError("Prediction name must contain only letters, digits, underscore, hyphen")
        predictor = OpenJevHTTP(args.endpoint, args.model_id, args.input_mode, args.timeout)
        paths = episodes(args.episodes)
        selected = {}
        for path in paths:
            if (path / f"{args.name}.jsonl").exists() or (path / f"{args.name}.meta.json").exists():
                raise ValueError("Prediction output exists; choose a new name")
            if args.eligible_only:
                if not (path / "complete.json").exists():
                    raise ValueError("Eligible-only replay requires complete episodes")
                source = path / "forecasts.jsonl"
                rows = read_jsonl(source) if source.exists() else []
                labels = label_forecasts(rows, read_jsonl(path / "oracle.jsonl"))
                selected[path] = {row["forecast_id"] for row in labels if row["label"] is not None}
        for path in paths:
            write_json(path / f"{args.name}.meta.json", {
                "mode": "offline_replay", "input_mode": args.input_mode,
                "model_id": args.model_id, "model_revision": args.predictor_revision,
                "endpoint": args.endpoint,
                "selection": "eligible_windows" if args.eligible_only else "all_windows",
            })
            source = path / "forecasts.jsonl"
            for row in read_jsonl(source) if source.exists() else []:
                if args.eligible_only and row["forecast_id"] not in selected[path]:
                    continue
                append_jsonl(path / f"{args.name}.jsonl", predictor.score(row))
    else:
        if not args.predictions or Path(args.predictions).name != args.predictions:
            raise ValueError("Predictions must be a filename stem")
        all_labels, all_predictions, per_episode = [], [], []
        group_rows, metadata, incomplete = {}, [], []
        for path in episodes(args.episodes):
            meta = json.loads((path / "episode.json").read_text())
            if not (path / "complete.json").exists():
                incomplete.append(meta["episode_id"])
                continue
            forecast_path = path / "forecasts.jsonl"
            forecasts = read_jsonl(forecast_path) if forecast_path.exists() else []
            labels = label_forecasts(forecasts, read_jsonl(path / "oracle.jsonl"))
            prediction_path = path / f"{args.predictions}.jsonl"
            predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
            prediction_meta = path / f"{args.predictions}.meta.json"
            # Online default uses the singular metadata filename.
            if args.predictions == "predictions" and not prediction_meta.exists():
                prediction_meta = path / "prediction.meta.json"
            if predictions and not prediction_meta.exists():
                raise ValueError("Missing prediction provenance: " + str(path))
            if prediction_meta.exists():
                metadata.append(json.loads(prediction_meta.read_text()))
            result = evaluate(labels, predictions, args.threshold)
            per_episode.append({"episode_id": meta["episode_id"], "scene": meta.get("scene_name"),
                                "metrics": result})
            all_labels.extend(labels)
            all_predictions.extend(predictions)
            group = (meta.get("pipeline", "unknown"), meta.get("scene_name", "").split("/")[-1])
            bucket = group_rows.setdefault(group, [[], []])
            bucket[0].extend(labels)
            bucket[1].extend(predictions)
        signatures = {json.dumps({k: m.get(k) for k in ("model_id", "model_revision", "input_mode", "mode", "selection")}, sort_keys=True)
                      for m in metadata}
        if len(signatures) > 1:
            raise ValueError("Cannot pool different prediction models/revisions/input modes")
        summary = evaluate(all_labels, all_predictions, args.threshold)
        summary.update({"episodes": per_episode, "incomplete_episode_ids": incomplete,
                        "prediction_metadata": metadata[:1],
                        "by_family_level": {"/".join(key): evaluate(*rows, threshold=args.threshold)
                                            for key, rows in group_rows.items()}})
        # Do not mix global and per-constraint rows in the headline metric.
        global_rows = [r for r in all_labels if r["constraint_id"] == "__all__"]
        global_ids = {r["forecast_id"] for r in global_rows}
        summary["global_task_forecast"] = evaluate(global_rows,
            [r for r in all_predictions if r["forecast_id"] in global_ids], args.threshold)
        write_json(args.output, summary)
        print(json.dumps({"report": str(Path(args.output).resolve()),
                          "episodes": len(per_episode), "incomplete": len(incomplete),
                          "global_metrics": summary["global_task_forecast"]["micro"]}, indent=2))


if __name__ == "__main__":
    main()
