# Action-conditioned safety Predictor Judge

The Predictor Judge receives adjacent overview/wrist observations, the current robot
state, the unexecuted suffix of the current execution segment, and a natural-language
constraint. It outputs a shared Noul `[No, Yes]` score for a **new violation during
that suffix**. Formal monitor states and AP truth are supervision, not model inputs.
It judges action-conditioned safety; it does not generate future actions, states or images.
The existing current-image classifier and agentic planner remain separate entrypoints.

## Use a prepared dataset

Use a recorded SafetyJev commit and its exact Open-Jev submodule:
`git submodule update --init --recursive`. The reference below uses the existing
released RGB observations and labels; no image regeneration or relabeling is needed.
Keep this dataset revision fixed throughout an experiment. Install the
[shared training environment](../README.md#training-environment) before the download
commands below.

The dataset is distributed through
[SafetyJev-Predictor-Judge](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/SafetyJev-Predictor-Judge).
Sign in with your own Hugging Face account. External access requires approval;
organization-member access follows the repository's gating settings.
Use a completed release: `DATASET_SUMMARY.json` must report `status: complete`.

From the **SafetyJev repository root**, download, verify and extract its independent ZIP parts:

```bash
.venv-visual/bin/hf auth login
.venv-visual/bin/hf download IDEAS-Lab-Northwestern/SafetyJev-Predictor-Judge \
  --repo-type dataset --revision c8d64200d3744a111eed9941f8b65e0a486403ea \
  --local-dir downloads/predictor_judge
.venv-visual/bin/python -c 'import json; assert json.load(open("downloads/predictor_judge/DATASET_SUMMARY.json"))["status"] == "complete"'
(cd downloads/predictor_judge && sha256sum -c SHA256SUMS)
(set -e
 for archive in downloads/predictor_judge/predictor-judge-*.zip; do
   unzip -oq "$archive"
 done)
```

The pinned release contains 532 episodes across six task families:

| Split | Episodes | Unsafe / safe | Positive / negative windows |
|---|---:|---:|---:|
| Train | 332 | 332 / 0 | 2,201 / 2,269,700 |
| Validation | 100 | 20 / 80 | 155 / 792,912 |
| Test | 100 | 31 / 69 | 168 / 874,240 |

Download size is approximately 213.6 GB in 31 ZIP parts; extracted raw data and
indices total 245.5 GB, excluding the cache, model and training outputs. Window
counts are not independent events. Training resamples the supplied pool; validation
and test retain their supplied distributions.

Every part contains paths beneath `datasets/predictor_judge/`; extract all parts
from the repository root, not from inside `datasets/`. Keep the dataset revision
returned by Hugging Face with the experiment record. Downloads can resume.
For limited disk space, download, verify and extract one part at a time, then
remove that downloaded ZIP; keep all extracted files.

Place the delivered directory at `datasets/predictor_judge/`:

```text
datasets/predictor_judge/
  README.md
  raw/<episode_id>/
  package/
    dataset_metadata.json
    train.jsonl
    validation.jsonl
    test.jsonl
    checksums/
```

Set `data.package` to `datasets/predictor_judge/package`. The package uses relative
paths into `raw/`; keep these two directories together. A delivered package already
contains window labels, frozen splits and training normalization. No resplitting
or relabeling is needed: continue with **Experiment tracking** and **Cache, train
and evaluate** below. Its dataset
README and composition report describe the exact included episodes. Inventory rows
for nonselected source episodes are provenance only and need not be downloaded.

Keep original dataset files under `datasets/`, disposable caches under
`datasets/cache/`, and training runs under `outputs/predictor_judge/training/`.
Preparation and transfer reports belong under `outputs/predictor_judge/preparation/`.
The existing classifier directories are independent and do not need to move.

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
  --split-manifest /path/to/group-splits.json --train-episodes unsafe_only
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
- By default, train uses only complete, valid, post-initialization **unsafe episodes**.
  Their nonviolating intervals and other constraints provide negative windows.
  Training-side safe episodes remain inventoried as a reserve. Missing or rare
  predicate violations are reported as observed, without requiring equal counts.
- To add safe training episodes explicitly, use `--train-episodes unsafe_plus_safe`.
  This adds up to `ceil(unsafe / 4)` active safe episodes per family (at most one if
  no unsafe episode exists); `--unsafe-per-safe` changes this optional ratio.
  Active means an arm joint range exceeds `--active-motion-rad` (default 0.05 radians),
  which does not establish object engagement or success.
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
reports the missing label, rather than fabricating balance. Final validation/test
use all eligible windows in their original proportions.

`sampling/epoch-*.json` reports the **planned** full-epoch composition.
`sampling-consumed.json` reconstructs actual consumed draws from the saved trainer
cursor, including positive/negative counts, unique samples, repeated draws and
distinct positive events. Both reports include available positive/negative windows,
positive events by family/constraint, and positive draws per available positive window.
Category balancing may repeat rare events more often; these reports distinguish
sampling exposure from the naturally uneven raw dataset.
Deterministic epoch/cursor reconstruction supports resume.
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

## Experiment tracking

Install the pinned visual environment in [training environment](../README.md#training-environment),
including the tracking extra when using W&B:

```bash
.venv-visual/bin/python -m pip install -e '.[tracking]'
export SAFETYJEV_TRACKING=wandb
export WANDB_ENTITY='<team>'
export WANDB_PROJECT='<experiment-project>'
export WANDB_NAME='<unique-run-name>'
export WANDB_TAGS='predictor-judge'
export WANDB_MODE=online
```

Authenticate W&B and Hugging Face on the training host using environment credentials
or their login tools. Repository configuration contains no account credentials.
`WANDB_MODE=offline` records SDK files for later synchronization;
`SAFETYJEV_TRACKING=none` uses local logs only.

Only rank zero opens a W&B run. `tracking.json` preserves its ID across resume;
do not set `WANDB_RUN_ID` or reuse an output directory for a different experiment.
Logs include loss, throughput, memory and validation metrics, including event recall
and false alarms. Full validation is logged separately from the diagnostic subset.
Numeric experiment parameters are logged; source data and filesystem paths are not
uploaded by this integration. W&B may collect its normal runtime metadata.
Startup authentication/SDK errors stop the run. Later telemetry failures warn while
training continues with local logs. The local `training.jsonl`, evaluation reports
and checkpoints remain the authoritative artifacts.

## Cache, train and evaluate

Use the environment in [training environment](../README.md#training-environment), from the repository root.
The classifier and Predictor Judge share the optimizer/DDP/checkpoint engine. The
Predictor Judge adds trainable state/action projections to the language LoRA and
scalar head; the base weights and vision encoder stay frozen.

Prepare the package above, then build its disposable training cache **on the server**:

```bash
.venv-visual/bin/python tools/prepare_predictor_judge_cache.py \
  --package datasets/predictor_judge/package --output datasets/cache/predictor_judge
.venv-visual/bin/python tools/prepare_predictor_judge_cache.py \
  --package datasets/predictor_judge/package --output datasets/cache/predictor_judge --verify-only
```

The SQLite cache deduplicates original PNG images and numeric/history windows
shared by constraint questions, and indexes JSONL offsets. It does not change labels,
resize the source images or duplicate the raw archive. Preparation checks raw record
and media identities and image hashes; reads check cached payload hashes. Incomplete
caches are refused. Re-running preparation resumes it; `--max-gib` applies periodic
budget checks with transaction headroom. Keep the package and referenced raw records
available. Build indices recursively from a campaign root with `build --episodes`;
duplicate episode IDs are rejected rather than silently counted twice.

The delivered archives do not contain this cache. Reserve space for both the
extracted dataset and a cache holding its referenced images plus database overhead,
as well as the model and training checkpoints. `DATASET_SUMMARY.json` reports the
download and extracted sizes; `--max-gib` can bound the cache's disk usage. If a
build stops, rerun the same preparation command to resume. Run `--verify-only`
successfully before starting training; a budget-stopped cache is not ready to use.

`configs/training/predictor_judge_27b_reference.json` specifies the pinned backbone,
three adjacent frames per camera, worker preprocessing, event-balanced training,
global batch 128 and an initial 100 optimizer updates. Its 12,800 training draws
use replacement; inspect distinct event counts as well as draw counts. This is
a starting experimental budget, not evidence of convergence.

Download the model once, before launching ranks/workers:

```bash
export HF_HOME="$PWD/checkpoints/hf_cache"
.venv-visual/bin/python - <<'PYMODEL'
import json
from huggingface_hub import snapshot_download
c = json.load(open('configs/training/predictor_judge_27b_reference.json'))
snapshot_download(c['model']['model_id'], revision=c['model']['revision'])
PYMODEL

.venv-visual/bin/python tools/visual_training_preflight.py \
  --config configs/training/predictor_judge_27b_reference.json --gpus 8
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv-visual/bin/python tools/profile_visual_loader.py \
  --config configs/training/predictor_judge_27b_reference.json \
  --workers 2 --batches 200 --output outputs/judge-loader-profile.json

NPROC_PER_NODE=8 bash scripts/train_predictor_judge.sh \
  --config configs/training/predictor_judge_27b_reference.json \
  --output outputs/predictor_judge/training/27b
```

The reference uses per-device batch 8, global batch 128 and two loader workers
per rank. The eight-rank command derives two gradient accumulation steps.
Its initial learning budget is 100 optimizer updates, with validation and saves
every 25 updates and 10 warmup updates. This is a starting experiment, not a
validated convergence recipe. At global batch 128 it consumes 12,800 draws; a
50% positive target yields approximately 6,400 positive draws. Review validation
and event exposure before extending the budget. Changing `samples_per_epoch`
alone does not reduce total draws: `max_steps` and global batch determine them.
Use `NPROC_PER_NODE=4` for four GPUs (four accumulation steps), or `1` for one
sufficiently large GPU (16 accumulation steps). Override `--batch-size`,
`--global-batch-size` and `--workers` as needed; accumulation is derived from
global batch / (GPUs × microbatch), which must be an integer.
Per-device batch 8 passed four-rank tests on pilot captures; eight-rank execution
and full-model training on the completed collection package remain to be validated.
DDP replicates the full backbone per GPU. This model processes up to six images,
versus the classifier's two, so memory and throughput must be measured separately.
GPU count does not change the data or model interface. See the shared guide for
hardware compatibility, resume rules and checkpoint layout.

The CPU workers prepare image/text tokens; numeric projections remain in the model
and receive gradients. A global deterministic sampling stream is partitioned across
ranks. Training pads draws to a global microbatch; evaluation partitions without
padding and merges predictions by sample ID. Duplicate IDs fail evaluation.

For an engineering smoke, use `predictor_judge_smoke.json` with the package path
set appropriately. It runs two updates with capped evaluation, not a quality test.
`--stop-after-step` supports controlled interruption. Resume in the same output:

```bash
NPROC_PER_NODE=8 bash scripts/train_predictor_judge.sh \
  --config configs/training/predictor_judge_27b_reference.json \
  --output outputs/predictor_judge/training/27b \
  --resume outputs/predictor_judge/training/27b/checkpoints/step-00000025
```

Resume requires matching model, data, source, training configuration and world size;
it restores optimizer, per-rank RNG and sampler cursor. Changing GPU count is a new
run, not an exact resume. Shared checkpoints are published by rank zero atomically.

During training, **all validation positive windows** plus a fixed uniform sample
of up to 4,096 negative windows select the checkpoint by diagnostic NLL.
`evaluation.validation_negative_samples` sets the negative budget; omit it or use
null for full validation. The former `validation_samples` key is rejected to avoid
silently applying a different sampling meaning. Selection runs once on rank zero;
indices, split hash and per-constraint population/selected counts are saved in
`validation-subset.json` and reproduced on resume. Each diagnostic report identifies
its sampled distribution: its precision and NLL are not population estimates.
Constraints with no positive events remain without recall evidence.
The reference configuration sets `evaluation.run_test=false`: development training
does not construct or iterate a test loader. Finalization evaluates the selected
model on **full validation**, without label balancing, and saves it in `final/model`.
The final report records `test_status: not_evaluated`. Setting `run_test=true`
explicitly also evaluates test on completion. A smoke-only `max_batches` cap can
truncate evaluation; keep it null for the reference experiment.

After choosing the experiment and checkpoint using validation, run the independent
full test once. Multi-GPU evaluation is supported:

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_model.sh --task predictor_judge \
  --checkpoint outputs/predictor_judge/training/27b/final/model \
  --package datasets/predictor_judge/package --frame-cache datasets/cache/predictor_judge \
  --split test --batch-size 8 --workers 2 \
  --output outputs/predictor_judge/training/27b/final-test.json
```

The evaluator checks complete split coverage and writes both a report and per-window
JSONL scores, including hashes of the model (numeric projections included), dataset
and split. Threshold defaults to 0.5; any alternative must be chosen on validation
and frozen before test evaluation. There is no separate calibration split.

## Export and publish the selected checkpoint

After full validation and the independent test, build the inference bundle:

```bash
.venv-visual/bin/python -m safetyjev.model_export build \
  --run outputs/predictor_judge/training/27b \
  --test-report outputs/predictor_judge/training/27b/final-test.json \
  --output outputs/predictor_judge/exports/reference

.venv-visual/bin/python -m safetyjev.model_export upload \
  --folder outputs/predictor_judge/exports/reference \
  --repo-id '<owner>/<model-repo>'
```

The shared exporter identifies the model type from its configuration. It requires
a completed run, uncapped validation and a full standalone test whose checkpoint,
dataset and split hashes match. The bundle contains:

```text
reference/
  model/          # language adapter, shared head, processor, model configuration
    predictor_judge.pt  # state/action projections and training normalization
  metrics.json    # validation/test window, constraint and event metrics
  provenance.json # experiment parameters and source/data hashes
  manifest.json   # exported file checksums
  README.md
```

The upload command verifies the bundle and creates a new private HF model repository.
Use a distinct repository for each experiment; existing repositories are not overwritten.
If upload fails after creation, inspect the destination and resume with the HF CLI.
Keep the returned revision with the experiment record. Optimizer/RNG state and
per-window prediction files remain in the local run; this bundle is for inference,
not exact training resume. Retain the local run until delivery is confirmed.

Download the published repository with `hf download '<owner>/<model-repo>'
--revision '<commit>' --local-dir '<destination>'`, then load `<destination>/model`
with `jev.predictor_judge_model.PredictorJudgeModel.load`. The loader obtains the
pinned frozen backbone separately; the artifact is not an AutoModel checkpoint.

Deliver the code/submodule/data revisions, actual training configuration, W&B run
link, HF model revision, validation/test reports and sampling-consumed report.
Keep full test predictions available for error analysis. None of these steps
requires integration with the agentic intervention loop.

## Interpret the evaluation

- **Window metrics:** confusion counts, violation recall/miss rate, false-positive
  rate, precision/F1, balanced accuracy, AUROC and average precision where defined,
  NLL and binary Brier. Reports partition these by constraint, family/constraint,
  and actual remaining length (1–8), and include an always-No accuracy baseline.
- **Event metrics:** each `(episode, constraint, first rejection)` is counted once.
  Recall asks whether at least one eligible positive window alerted before it;
  lead steps use the earliest such alert. Only events with eligible positive windows
  are in this denominator. Multiply steps by the recorded control interval for time.
- **Safe-episode false alarms:** fraction of wholly safe source episodes with at least
  one alert. Interpret only on a full uncapped split, not the development subset.
- **Coverage/composition:** raw selection, excluded-label reasons, available events,
  actual sampled draws and repeated samples remain visible in package/run reports.

Predictor Judge `micro.brier` is mean `(p_yes - y)^2`; the classifier's upstream
`brier` sums both No/Yes squared errors and is twice that quantity for binary labels.
The optimization loss is shared. `answers` additionally reports No/Yes support and
macro F1 consistently with the classifier. A missing class has undefined recall;
large counts of adjacent windows are not independent events. Judge Yes means a new
violation; classifier Yes answers its particular AP question and can mean safety.

Training-balanced scores are not established deployment probabilities. No runtime
intervention or speed benefit is claimed by these offline metrics. Forward timing
includes tensor transfer but excludes worker preprocessing and data waiting; inspect
training input-wait and global-samples/s logs for throughput. The initial experiment
should examine event recall and safe-operation false alarms alongside per-constraint
metrics, rather than judging quality from training loss or pooled accuracy alone.
