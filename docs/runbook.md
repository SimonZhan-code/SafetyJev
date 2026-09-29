# Run the first shadow evaluation

All simulator commands below are for a Linux/NVIDIA machine with ManiGuard's
BEHAVIOR/OmniGibson assets, valid Spot+Buddy, and the long-finger Franka asset.
The Mac workspace can run the CPU evaluation tests. No remote node is assumed.
Follow the [upstream setup](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
and [evaluation instructions](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/run_benchmark/).

Use separate environments for openpi policy serving, the behavior simulator, and
Open-Jev. Install this repository into the behavior environment with `pip install
-e /path/to/SafetyJev`, or put its root on PYTHONPATH. Paths below are deployment
examples; substitute the actual directories.

## 1. Pin code and resources

Use a separate ManiGuard checkout at commit
`be97624e0acbec6b6f9260a08891b04168eb8e6c`. The adapter checks hashes of the runner
and monitor implementation. It instruments a copy in memory, not files on disk.

```bash
python -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard
hf download IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora \
  --revision 1d84eda070313a202a595e449fcf41a1a1e8a546 \
  --include '7400/*' --local-dir /data/checkpoints/pi05-jar
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
  --revision 2ea32a1451669fb736ae78ffce9cc82aad4cceac \
  --include 'jar_transport/*' --local-dir /data/maniguard-bench
```

These commands download real, potentially large resources; they were not executed
as part of local implementation. Public API metadata, including the available
policy steps 1850/3700/5550/7400, was checked on 2026-09-28.

## 2. Serve π0.5 in the openpi environment

From the configured ManiGuard checkout:

```bash
python -m maniguard.serve.openpi_native \
  --config pi05-base_datagen_v1_jar_joint_2cam_lora \
  --checkpoint /data/checkpoints/pi05-jar/7400 --port 8000
```

This released snapshot uses the native openpi/Orbax path; do not assume it is a
standalone PyTorch checkpoint. Preserve the jar configuration's left overview +
wrist cameras, absolute joint controller, assisted grasping, and execute horizon.
Record the actual loaded checkpoint and source versions. Our provenance JSON is
a declaration, not remote attestation of the policy server's loaded weights.

## 3. Capture one unchanged-policy rollout

From the ManiGuard root, in the behavior environment:

```bash
python -m safetyjev.cli capture \
  --maniguard-root /path/to/ManiGuard \
  --output /data/safetyjev/jar-smoke \
  --provenance /path/to/SafetyJev/configs/jar-provenance.json \
  -- \
  --config configs/eval/jar_transport_joint.yaml \
  --benchmark-root /data/maniguard-bench/jar_transport \
  --scenes task_0000/base --seed 0 --tag safetyjev-shadow
```

The default records predictions at chunk starts. Add `--recheck-every 1` BEFORE
the `--` separator for remaining-chunk windows after every action. This changes
the number of forecasts, not the executed policy horizon. Keep experiments separate.

Every episode gets a UUID directory:

```text
episode.json                provenance, task spec, execution config
forecasts.jsonl             one pre-action input per window × constraint
frames/                    current overview and wrist PNGs
oracle.jsonl               per-step raw monitor outcomes; evaluator-only
labels.jsonl               binary or excluded/censored window labels
maniguard_result.json       original success/safety/engagement outcome
complete.json               end marker
```

No model is needed to collect the first forecast dataset. This mode still runs
real π0.5 and ManiGuard; it is not a mock simulator. Capture failures and simulator
crashes must be inspected. The report lists incomplete captured episodes but cannot
count scene-load failures that happen before capture starts: inspect ManiGuard's
original results.jsonl and retain the complete planned-scene manifest as well.

## 4. Score the saved inputs

Serve Zefan-Cai/Open-Jev with its own loader and the pinned published
`ZefanCai/Open-Jev-2B` package revision
`0c7aa498b1627be8da4acf34c863ff0ee0a92785`. Its model card supplies the exact
upstream Qwen revision and launch instructions. The package includes an adapter
and head, not the base weights. Example once the package is downloaded:

```bash
python -m jev.server --checkpoint /data/checkpoints/open-jev-2b/package/checkpoint \
  --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8791
python -m safetyjev.cli predict --episodes /data/safetyjev/jar-smoke \
  --name openjev2b-proprio \
  --endpoint http://127.0.0.1:8791/v1/systemone \
  --model-id Open-Jev-2B \
  --predictor-revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 \
  --input-mode proprio_only
```

This explicitly omits camera pixels. It cannot establish the quality of a
multimodal safety model. Scores from a future multimodal scorer should use the
same `forecast_id`, with JSONL rows:

```json
{"forecast_id":"episode:step:constraint", "score":0.3, "error":null, "latency_s":0.08}
```

Store `<name>.meta.json` beside them with `model_id`, `model_revision`, `input_mode`,
and `mode`. Resolve image paths relative to the episode directory. Supply only
the forecast's `input`, never `oracle.jsonl`, labels, or episode outcomes.
The illustrative score above is not a model result.

For online synchronous shadow scoring, instead add
`--online-predictor /path/to/SafetyJev/configs/openjev-state-only.json` to capture
before the separator. Each request finishes before env.step. Timeouts are recorded
as prediction failures; no prediction is used to alter actions.

## 5. Report prediction quality

```bash
python -m safetyjev.cli report --episodes /data/safetyjev/jar-smoke \
  --predictions openjev2b-proprio --threshold 0.5 --output /data/safetyjev/report.json
```

The report has a separate `global_task_forecast` block for comparison with the
combined ManiGuard monitor, plus per-constraint, episode, and family/level detail.
Use the same captures and a new prediction name for a fine-tuned checkpoint.
Existing predictions are not overwritten; mixed model revisions are rejected.

Before claiming performance: verify real source-hook execution and matched-seed
action parity, validate monitor coverage, establish held-out task groups, run both
untuned and tuned scorers on identical windows, and show positive-event counts.
No positive events means the pilot cannot estimate violation recall.
