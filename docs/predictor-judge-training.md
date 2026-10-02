# Action-conditioned safety Predictor Judge

The Predictor Judge receives adjacent overview/wrist observations, the current robot
state, the unexecuted suffix of the current execution segment, and a natural-language
constraint. It outputs a shared Noul `[No, Yes]` score for a **new violation during
that suffix**. Formal monitor states and AP truth are supervision, not model inputs.
It judges action-conditioned safety; it does not generate future actions, states or images.
The existing current-image classifier and agentic planner remain separate entrypoints.

## Capture raw rollouts

Capture uses a ManiGuard checkout with passive recording API 1. The collector does
not query Jev, replace actions or change the policy's execution horizon. Run it with
the simulator's Python environment after starting the desired policy server.

Create a provenance JSON containing the checkpoint repository, exact revision and
step, benchmark revision, and purpose of the collection. These values must describe
the policy actually loaded by the server; retain its startup log with the campaign.

```bash
python -m safetyjev.predictor_judge_commands capture \
  --maniguard-root /path/to/ManiGuard \
  --output datasets/predictor_judge/raw \
  --provenance /path/to/capture-provenance.json -- \
  --config configs/eval/clutter_pickup_joint.yaml \
  --benchmark-root /path/to/maniguard-bench/clutter_pickup \
  --scenes task_0034/base --seed 0 --port 8000 --max-steps 1100
```

Benchmark arguments are resolved from the ManiGuard checkout. Recording output and
provenance paths are resolved from the command's original working directory.
Supported action convention: seven absolute arm joint targets in radians plus the
binarized gripper command, without IK conversion, with execute horizon at most eight.

Each episode directory contains:

- `episode.json`, `initialization.json`: task, source, policy and control metadata.
- `observations/`: lossless RGB PNGs; `observations.jsonl` indexes them by control step.
- `states/`: chunked NPZ state arrays and per-step physical measurements, object states,
  contact pairs, camera poses/calibration and assisted-grasp information. Calibration
  includes USD projection parameters and image resolution; an unready intrinsic
  matrix is marked unavailable rather than saved as valid zero focal lengths.
- `proposals.jsonl`: full policy replies and the canonical prefix planned for execution.
- `attempts.jsonl`, `execution.jsonl`, `events.jsonl`: submitted commands, commands whose
  environment step returned, and observations/termination status. A failed observation
  does not erase an action that was already applied.
- `oracle.jsonl`: per-step AP values, per-constraint rejection states and available
  predicate measurements, including liquid baselines and contained-particle counts.
- `snapshots/`: query-boundary scene/robot/controller/particle snapshots and monitor/RNG
  state for recorded-command replay. The remote policy server's RNG is not included.
- `record.json`: compact episode index consumed by the package builder.
- `episode_status.json`: completion or failure state, with pending attempts retained.
- `recording_timings.json`: observer callback costs, separate from policy and simulation time.

Snapshots require the matching scene/assets and simulator to restore. They are not a
promise of bitwise replay. Base scene copies are saved as `task_scene.json` and
`task_diagnostics.jsonl` when available. Camera PNG bytes remain the training observation
source; preview MP4 playback FPS is not the control clock.

## Construct the training package

```bash
python -m safetyjev.predictor_judge_commands build \
  --episodes datasets/predictor_judge/raw \
  --output datasets/predictor_judge/package --history-frames 3 --seed 42 \
  --split-manifest /path/to/group-splits.json
```

Freeze `group-splits.json` before reviewing collection outcomes. It maps base-task
IDs such as `jar_transport/task_0000` to `train`, `validation` or `test`. All conditions,
policies and seeds of a base task stay together. Omitting the manifest enables an
80/10/10 engineering convenience split, not a frozen experimental protocol.
There is no calibration split.

Collection and training admission are separate:

- Keep every attempt in the raw archive, including failures. The builder inventories
  incomplete episode directories when their metadata exists. Failures before the
  recorder starts must also be retained in the campaign's attempt ledger.
- Complete, valid, post-initialization unsafe episodes form the training core.
  Add up to `ceil(unsafe / 4)` active safe episodes per family. If no unsafe episode
  exists for a family, retain at most one active safe episode and report the shortage.
  This is an approximate 4:1 admission target, not a claimed observed prevalence.
- Active safe selection uses a disclosed heuristic: maximum range of an arm joint
  exceeds 0.05 radians (`--active-motion-rad`). It does not prove object engagement
  or success. `--unsafe-per-safe` adjusts the ratio.
- Validation/test keep **all valid episodes** from their assigned groups, including
  safe and inactive ones. They are never balanced by outcome. Initially violated,
  incomplete, invalid-GT or invalid-media attempts remain listed with reasons.
- A failed episode with a witnessed violation is still inventoried as unsafe, but
  is quality-excluded from the finalized package. Complete-prefix labels could be
  recovered separately; this conservative admission does not relabel them safe.

The builder writes `episode_inventory.jsonl` with all episode decisions,
`composition.json` with episode/event/window counts and `DATA_SUMMARY.md` for readers.
It does not delete or copy raw episodes. Membership in these manifests defines the
curated raw package; the collection directory alone is not the training dataset.

`train.jsonl`, `validation.jsonl` and `test.jsonl` contain eligible window indices;
`excluded.jsonl` retains unusable windows **within admitted episodes**. Whole-episode
exclusions are in the inventory. A single unsafe episode also supplies negative
windows for other constraints and nonviolating intervals. Normalization is fitted
only on selected training episodes.

`dataset_metadata.json` binds resource paths, hashes, groups, selection parameters,
counts and normalization. Per-episode media manifests verify image bytes. Keep raw
files and the package together when transferring data, preserving relative paths;
the indices alone are not a standalone dataset. No family/policy/safety metadata
used for selection or sampling is passed to the model.

### Training draws

The train loader targets 50% positive draws when both labels exist. It selects a
family/constraint, then a distinct episode/constraint/first-rejection event, then
one of that event's windows. Negatives are sampled by family/constraint, safe or
unsafe source, near-event or ordinary interval, and episode before window.
A near-event negative ends within eight steps **before** its constraint's first
rejection and must retain a complete negative GT label. Temporal proximity alone
does not establish perceptual difficulty.

The default draw budget is twice the positive window count at a 50% target. Set
`data.sampling.samples_per_epoch` for an explicit budget, and
`data.sampling.positive_fraction` to change the target. Sampling uses replacement;
more draws do not create new events. A single-class pool remains single-class and
reports the missing label, rather than fabricating balance. Validation/test use
all eligible windows in their original proportions.

`sampling/epoch-*.json` reports the **planned** full-epoch composition.
`sampling-consumed.json` reconstructs actual consumed draws from the saved trainer
cursor, including positive/negative counts, unique samples, repeated draws and
distinct positive events. Deterministic epoch/cursor reconstruction supports resume.
Model selection uses validation; a score trained under rebalanced sampling is not
a claim of calibrated deployment probability.

## Input and supervision contract

For `F=3`, each camera batch is `(B,3,3,H,W)` uint8: current frame and the preceding
two adjacent control frames. Missing startup history is explicitly masked. No future
frames, frame skipping or past executed actions enter the predictor judge.

The current state has 16 features: arm joint positions (7), arm joint velocities (7),
mean gripper position (1) and mean gripper velocity (1). Arm units are radians and radians/second; gripper units are metres and metres/second. Exact feature names are in
`state_features`; the raw archive also keeps the policy's original state vector.

Actions are `(B,8,8)` with Boolean `(B,8)` masks. Five remaining commands occupy the
first five positions; the last three are masked, not zero/hold commands to execute.
Their real prediction horizon is five control intervals, not eight. The numeric
projection and multimodal attention use the same validity mask.

At state t, action t leads to state t+1. A suffix of h commands is labeled from oracle
samples t+1 through t+h. A witnessed new violation is Yes. A fully observed interval
without rejection is No. Early termination without a witnessed violation is censored;
a constraint already rejected at t is not a new-violation target. Temporal constraints
remain present with their original GT semantics. Short observations may omit relevant
past events; no privileged monitor memory is silently added to compensate.

## Train and evaluate

Use the training environment described in [visual training](visual-training.md).
Set the package path and training budget in a copy of
`configs/training/predictor_judge_smoke.json`. That configuration is a two-update engineering
smoke, not a scientific training recipe.

```bash
python -m safetyjev.predictor_judge_train \
  --config configs/training/predictor_judge_smoke.json \
  --output outputs/predictor-judge-training --device cuda:0

python -m safetyjev.predictor_judge_eval \
  --checkpoint outputs/predictor-judge-training/final/model \
  --package datasets/predictor_judge/package --split test \
  --output outputs/predictor-judge-test.json --device cuda:0
```

Resume in the original output directory with `--resume` pointing to its latest
`checkpoints/step-XXXXXXXX`. Source, data and training identity must match. Model
selection uses validation NLL; the selected checkpoint is evaluated on test after
training. The reference trainer is single-device and preserves the existing scalar
head/language-LoRA/frozen-vision convention, adding trainable state/action projections.

Reports include confusion counts, precision/recall, AUROC/average precision where
defined, Brier/NLL, per-constraint and per-valid-length breakdowns, and excluded-label
counts. Adjacent windows are correlated; window counts are not independent trials.
Timing covers processor/model batches, not a real-time simulator/control loop.
