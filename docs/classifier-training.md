# Classifier training

For a shared training delivery, point the existing package/cache commands at
`DATASET_ROOT/classifier`. See the [portable delivery layout](data-preparation.md#portable-delivery-across-source-nodes);
the package catalog supplies the unified splits and train-only normalization.
A training-media delivery includes the original compressed input images under
`DATASET_ROOT/episodes/<family>/<run>/`; both models share these resources.
The separate raw backup is not required. Set
`data.frame_cache` to `null` to train directly from the delivered JPEGs, or build
an optional input cache on the training machine later.
For semantic training, `data.balance: "query_answer"` with
`data.positive_fraction: 0.6` assigns 60% of sampling probability to Yes when
both labels exist. This changes only training sampling; validation/test retain
their natural distributions and labels remain unchanged.

The classifier receives the current overview image, wrist image and a
natural-language AP question, and returns `[No, Yes]` scores. It retains the native
visual encoder and multimodal fusion of the Open-Jev fork, freezes the backbone,
and trains language LoRA plus a shared scalar head. Yes answers the question;
it is not universally an unsafe label.

This guide covers data download, preprocessing, training, evaluation and model
delivery. The action-conditioned model has its own [Predictor Judge guide](predictor-judge-training.md).

For source-derived semantic packages, follow the [semantic build and cache sequence](data-preparation.md#semantic-safety-labels-from-source-recordings)
instead of the legacy AP/video preparation below. In that explicit mode Yes means
the queried violation is present; single-frame liquid interval queries are excluded.
Use the corresponding semantic package/cache and `evaluation.run_test=false` for
development checks. The model and training entrypoint are unchanged.

## Unsafe600 start to finish

This is the current release workflow. First follow the shared
[code/environment setup](../README.md#training-environment) and
[pinned Unsafe600 download](../README.md#download-the-current-training-dataset).
The legacy dataset/build instructions later in this guide are separate workflows.
No simulator, raw download or semantic construction is needed here.

### Prepare a run and a separate smoke test

From the repository root, generate local configs. The formal reference budget is
an initial recipe, **not a validated convergence budget**; choose `max_steps`,
evaluation/save intervals and global batch after the pilot and record the choices.
The classifier recipe filename retains its historical name but the generated
config below explicitly uses the new six-family semantic package.

```bash
export HF_HOME="${HF_HOME:-$PWD/checkpoints/hf_cache}"
.venv-visual/bin/python - <<'PYCONFIG'
import copy,json
from pathlib import Path
from huggingface_hub import snapshot_download
out=Path('outputs/unsafe600-config');out.mkdir(parents=True,exist_ok=True)
c=json.loads(Path('configs/training/five_family_visual_27b_reference.json').read_text())
c['data'].update(package=str(Path('datasets/unsafe600/classifier').resolve()),
                 frame_cache=None)
c['data'].update(balance='query_answer',positive_fraction=0.6)
c['evaluation']['run_test']=False
c['evaluation']['max_batches']=None
(out/'classifier.json').write_text(json.dumps(c,indent=2)+'\n')
pilot=copy.deepcopy(c)
pilot['data']['batch_size']=1
pilot['training'].update(max_steps=4,global_batch_size=8,warmup_steps=1,
                         eval_every=2,save_every=2)
pilot['evaluation']['max_batches']=1
(out/'classifier-pilot.json').write_text(json.dumps(pilot,indent=2)+'\n')
snapshot_download(c['model']['model_id'],revision=c['model']['revision'])
PYCONFIG
```

Inspect GPU count/VRAM and free disk before loading the model. Set
`NPROC_PER_NODE` explicitly for the actual host (1 for one GPU, 8 for eight GPUs).
This launcher uses DDP: each GPU must fit the model; GPU memory is not pooled.
Global batch must be divisible by microbatch times GPU count. The pilot assumes
1, 2, 4 or 8 ranks; adapt its global batch for other counts. Keep the Judge's images,
history/state/action fields and masks intact; reduce microbatch if necessary.

Use a **separate pilot output** and retain its reports as interface diagnostics:

```bash
export NPROC_PER_NODE=8  # change to the actual supported host configuration
bash scripts/train_visual.sh \
  --config outputs/unsafe600-config/classifier-pilot.json \
  --output outputs/unsafe600-classifier-pilot --stop-after-step 2

bash scripts/train_visual.sh \
  --config outputs/unsafe600-config/classifier-pilot.json \
  --output outputs/unsafe600-classifier-pilot \
  --resume outputs/unsafe600-classifier-pilot/checkpoints/step-00000002
```

Accept the pilot only after real forward/backward updates, finite loss,
checkpoint save/resume and final model loading succeed. Check the actual batch,
mask/token lengths, GPU peak memory and validation output. Four updates and capped
validation are not model-quality evidence. The delivered dataset batch checks do
not replace this GPU/optimizer/DDP check. Keep config/code/world size unchanged
across the pilot resume.

### Train, evaluate and hand over

Configure optional W&B using the tracking section below. Keep credentials and
account destinations on the host. In a persistent terminal session, launch a new
formal output after selecting the budget; do not resume the pilot as a formal run:

```bash
bash scripts/train_visual.sh \
  --config outputs/unsafe600-config/classifier.json \
  --output outputs/unsafe600-classifier

NPROC_PER_NODE=8 bash scripts/evaluate_model.sh --task classifier \
  --checkpoint outputs/unsafe600-classifier/final/model \
  --package datasets/unsafe600/classifier --split validation \
  --batch-size 8 --workers 2 --output outputs/unsafe600-classifier/validation.json
```

Adjust evaluator rank count to the host as well. `final/model` is the trainer's
validation-selected export. Verify it with standalone full-validation inference.
After model/threshold selection, run the same independent evaluator with
`--split test` and a distinct output. Training keeps test disabled; evaluation
uses natural distributions. A capped pilot report must not enter the formal report.
Report accuracy, recall, precision/confusion counts and measured inference duration;
keep detailed diagnostics in JSON. Retain code/submodule/model/data revisions,
resolved config, seed, sampling report, selected checkpoint, logs and metrics.
Use the export section below when model publication is authorized.

The semantic Classifier asks whether the queried violation is present now.
It has no future-chunk/ordered shadow-evaluation role. Current-frame liquid-loss
queries are excluded because they require interval evidence. The formal package
contains 5,765,453 eligible windows; do not substitute the legacy AP counts below.

## Environment

Install the [shared training environment](../README.md#training-environment).
Use the same environment for data preparation, training, evaluation and export.
Commands below run from the repository root.

## Fixed inputs

Use a recorded SafetyJev commit and its exact Open-Jev submodule commit. Run
`git submodule update --init --recursive` after checkout. Do not update the
submodule independently during an experiment.

Download the five-family dataset from
[SafetyJev-Data](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/SafetyJev-Data)
with an HF account that has repository access:

```bash
.venv-visual/bin/hf auth login
.venv-visual/bin/hf download IDEAS-Lab-Northwestern/SafetyJev-Data safetyjev-five-family.zip \
  --repo-type dataset --revision c184b83f2765082a37174040be88d6c89b7a515a \
  --local-dir downloads
unzip downloads/safetyjev-five-family.zip
```

Run from the **SafetyJev repository root**. The ZIP already contains `datasets/`;
do not extract it inside another `datasets/` directory. It contains 4,816 episodes,
9,632 videos and 3,113,929 image/question pairs (about 25.3 GB extracted): Jar,
Lid, Stack, Cabinet and Dusty.

```text
datasets/
  README.md                   # delivered dataset details
  raw/                        # videos and AP traces
  annotations/                # trajectory GT
  definitions/                # AP question definitions
  packages/five_family/
    train.jsonl
    validation.jsonl
    test.jsonl
    dataset_metadata.json
```

The archive includes prepared indices; rebuilding is optional and documented in
[data preparation](data-preparation.md). Validate the delivered package:

```bash
.venv-visual/bin/python tools/check_data_pipeline.py \
  --package datasets/packages/five_family --output outputs/data-check.json
```

Keep the supplied base-task-group split for the first reference experiment:

| Split | Episodes | Pairs |
|---|---:|---:|
| Train | 3,781 | 2,440,945 |
| Validation | 534 | 334,276 |
| Test | 501 | 338,708 |

Changes to splits, question wording, sampling, model or training budget define
another experiment and must be recorded. Keep all conditions, policies and
seeds of one base task together. Do not rebalance validation or test.

## Optional input cache

Training-media deliveries can train directly with `data.frame_cache=null`. If you
choose to prepare a cache, build and verify it from the repository root:

```bash
.venv-visual/bin/python tools/prepare_visual_cache.py \
  --package datasets/packages/five_family --output datasets/cache/five_family --workers 8
.venv-visual/bin/python tools/prepare_visual_cache.py \
  --package datasets/packages/five_family --output datasets/cache/five_family --verify-only --workers 8
```

Preparation indexes every split and sequentially decodes each needed source
video. `--workers 8` processes up to eight videos concurrently (default: one);
reduce it on hosts with less CPU or RAM. Workers buffer at most one video each,
and one writer preserves the same lossless cache and resume checks. Identical video/frame references shared by AP questions are stored once
as lossless PNG records in `frames.sqlite`. No millions-of-files extraction and
no repeated MP4 seeking during training. `cache.json` marks completion; incomplete
caches are refused by the loader. Rerun the same preparation command to resume.
Full verification checks the database hash, SQLite integrity, and every PNG hash
and decoded image. `--verify-only --workers 8` uses eight CPU processes and
sequential row blocks without changing the cache. Read `verification.json` for
the current stage, verified-frame count, total count, and elapsed time.
Read `progress.json` for live stage/frame counts; do not scan the SQLite database
during preparation, since long read transactions can block writer commits.
`--max-gib` stops a growing build at periodic disk-budget checks and leaves it
incomplete; allow transaction headroom. Full preparation is intended for the
training server's NVMe, not a duplicate local dataset release.

The reference config uses this cache and CPU-worker multimodal preprocessing.
Only images and question text enter the model; labels and source metadata remain
outside the input tensors. Download the pinned model once before starting workers
or distributed ranks:

```bash
export HF_HOME="$PWD/checkpoints/hf_cache"
.venv-visual/bin/python - <<'PYMODEL'
import json
from huggingface_hub import snapshot_download
c = json.load(open('configs/training/five_family_visual_27b_reference.json'))
snapshot_download(c['model']['model_id'], revision=c['model']['revision'])
PYMODEL

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv-visual/bin/python tools/profile_visual_loader.py \
  --config configs/training/five_family_visual_27b_reference.json \
  --workers 2 --batches 200 --output outputs/loader-profile.json
```

Compare 0/2/4 workers on the actual host. The profile includes worker startup and
reports subsequent batch-wait percentiles; loader-only speed is not GPU training
throughput. Each training update separately records input wait, update time and
global samples/s. Warm up before interpreting them. With worker preprocessing,
evaluation forward timing includes tensor transfer and model computation but
excludes worker preprocessing; it is not end-to-end deployment latency.

## Optional W&B logging

Install the optional tracking extra in the same environment:

```bash
.venv-visual/bin/python -m pip install -e '.[tracking]'
```

Set runtime values for the experiment. Account/project names do not belong in
shared training configuration files. Authenticate on the host; never include
credentials in a config, command argument or handoff document.

```bash
export SAFETYJEV_TRACKING=wandb
export WANDB_ENTITY='<your-team>'
export WANDB_PROJECT='<experiment-project>'
export WANDB_NAME='<unique-run-name>'
export WANDB_TAGS='classifier,five-family'
export WANDB_MODE=online   # offline also supported; sync SDK files afterwards
```

Only rank zero starts a run. `tracking.json` stores its generated run ID and
project/entity; resuming the original output directory reuses them. Do not set
`WANDB_RUN_ID` or share an output directory between concurrent runs. A restarted
attempt is recorded separately on the same optimizer-step axis: after a crash,
uncheckpointed steps can reappear. Local `training.jsonl` is authoritative.
Offline restarts produce SDK segments that must all be synced; online identity
and permissions should be checked on the deployment host.

Logs include numeric training/validation metrics, throughput and rank-zero peak
allocated GPU memory. No observations, raw trajectories or full config dumps are
sent. Automatic source-code/git and console capture are disabled. W&B may still
record its normal runtime metadata. Missing SDK/authentication at startup is an
error; a subsequent telemetry exception warns and stops telemetry while local
training continues. Set `SAFETYJEV_TRACKING=none` to train without the SDK.

## Prepare and verify the host

Run the preflight before testing the actual target model on all intended GPUs.
A small-model test alone does not validate full-model memory or NCCL.

```bash
.venv-visual/bin/python tools/visual_training_preflight.py \
  --config configs/training/five_family_visual_27b_reference.json --gpus 8
```

For a short functional check, the small-model configuration runs four updates
with capped evaluation:

```bash
bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-smoke --device cuda:0
```

This checks the training path, not model quality. On the target hardware, test
training, checkpoint restore and validation with the intended model:

Use an output directory distinct from formal training:

```bash
NPROC_PER_NODE=8 bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_27b_reference.json \
  --output outputs/visual-training/deployment-check --stop-after-step 10

NPROC_PER_NODE=8 bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_27b_reference.json \
  --output outputs/visual-training/deployment-check \
  --resume outputs/visual-training/deployment-check/checkpoints/step-00000010 \
  --stop-after-step 20
```

These pause at step boundaries; they do not complete the 1,000-update budget.
Check finite loss, input wait, optimizer-step time, memory, checkpoint load and
successful continuation to step 20. Verify the evaluation path on validation:

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_model.sh --task classifier \
  --checkpoint outputs/visual-training/deployment-check/checkpoints/step-00000020/model \
  --package datasets/packages/five_family --frame-cache datasets/cache/five_family \
  --split validation --batch-size 8 --workers 2 \
  --output outputs/visual-training/deployment-check/validation-check.json
```

This standalone evaluation is complete, not a capped smoke metric. It verifies
distributed sample coverage and records checkpoint and data hashes.

## Reference experiment

The common configuration is `configs/training/five_family_visual_27b_reference.json`:
per-device batch 8, global batch 128, two data workers per rank, BF16, LoRA rank 8,
validation/save every 100 updates, and a starting budget of 1,000 updates.
Eight ranks derive accumulation 2; four ranks derive 4. Set `NPROC_PER_NODE`,
`--batch-size`, `--global-batch-size` and `--workers` to fit the host. More GPUs
replicate the model; they do not reduce its per-device memory requirement.

| GPUs | Per-device batch | Accumulation | Global batch |
|---|---:|---:|---:|
| 8 | 8 | 2 | 128 |
| 4 | 8 | 4 | 128 |
| 1 | 8 | 16 | 128 |

Accumulation must be an integer. Worker counts are per rank; an eight-rank run
with two workers per rank creates sixteen data workers. Measure throughput and
memory before increasing batch size. The reference uses BF16, gradient
checkpointing, AdamW with warmup/cosine learning rates, gradient clipping and the
Open-Jev cross-entropy plus Brier objective.

Training ranks partition one deterministic weighted draw stream. Evaluation uses
disjoint partitions without padding, so each selected sample is counted once.
A failed rank fails the run; a partial result is not a complete evaluation.

The budget draws 128,000 pairs with weighted replacement and is **not one epoch**
or a convergence claim. Training samples inverse family/question/answer frequency;
rare events may be repeated frequently. Judge learning using validation support
and per-question metrics, not only training loss or nominal pair count.

```bash
NPROC_PER_NODE=8 bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_27b_reference.json \
  --output outputs/visual-training/reference-run
```

Training selects its checkpoint by NLL on a fixed recorded 4,096-row validation
subset. Full validation is evaluated at completion. `evaluation.run_test=false`
in this configuration keeps test evaluation separate. Exact resume requires the
same world size, configuration (including total budget), code and data. Do not
increase `max_steps` and describe it as an exact resume; agree a new experiment
budget before launching it.

## Checkpoints and resume

```text
run/
  run.json                       # configuration and source/data identity
  training.jsonl                 # local step metrics
  tracking.json                  # W&B run identity when enabled
  checkpoints/step-00000100/
    model/                       # processor, language adapter, head, model config
    training_state.pt            # optimizer, RNG, trainable weights and cursor
    checkpoint.json              # identity and artifact hashes
  evaluations/                   # validation and standalone test outputs
  final/
    model/                       # validation-selected checkpoint
    report.json
```

Resume from the latest retained checkpoint in the original output directory,
using `--resume <run>/checkpoints/step-XXXXXXXX`. Snapshots are written at optimizer
step boundaries and restore the optimizer, per-rank RNG and epoch/batch cursor.
Model, data, source, configuration and GPU world size must agree. Changing those
settings starts a new experiment. Frozen base weights are restored from the pinned
revision rather than duplicated in each checkpoint.

Rank zero writes shared checkpoints; final model/reports are published atomically.
An interrupted finalization can be retried from the latest training checkpoint.
Reference runs use deterministic Torch/cuBLAS settings, but cross-hardware bitwise
equality is not assumed. Preserve training snapshots locally for exact resume;
the HF inference export below does not contain optimizer state.

## Final test and interpretation

After the experiment/checkpoint is selected using validation, run test once:

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_model.sh --task classifier \
  --checkpoint outputs/visual-training/reference-run/final/model \
  --package datasets/packages/five_family --frame-cache datasets/cache/five_family \
  --split test --batch-size 8 --workers 2 \
  --output outputs/visual-training/reference-run/final-test.json
```

Reports include per-question and pooled confusion counts, No/Yes support,
precision/recall/F1, two-class macro F1, balanced accuracy, NLL and Brier. No/Yes
are answers to the question: **Yes is not universally unsafe**. A missing class
has undefined recall; do not replace it with a made-up success or failure rate.
Inspect per-question results alongside the pooled scores. The data consist of
reviewed unsafe episodes, including their normal frames; they cannot establish
the false-alarm rate over a natural population of wholly safe episodes.

Test must not select the training budget, hyperparameters or decision threshold.
The classifier evaluator uses the fixed 0.5 No/Yes threshold and temperature 1.
Keep test predictions (`final-test.jsonl`) with the full local run for analysis.

## Export and deliver

Build a local bundle only after full validation and full standalone test:

```bash
.venv-visual/bin/python -m safetyjev.model_export build \
  --run outputs/visual-training/reference-run \
  --test-report outputs/visual-training/reference-run/final-test.json \
  --output outputs/visual-training/reference-export
```

The exporter checks that test data and model hashes match the selected run,
refuses capped training evaluations, and creates:

```text
reference-export/
  model/          # processor, language adapter, shared head, pinned model config
  metrics.json    # full validation/test aggregate and per-question metrics
  provenance.json # experiment parameters and data/source hashes
  manifest.json   # exported file checksums
  README.md
```

The optimizer, RNG snapshots, source data and per-sample test predictions stay
in the local run. Local filesystem paths in training reports are not exported.
The pinned frozen backbone is downloaded separately when loading `model/` with
`jev.visual_model.VisualDecisionModel.load`. This is not an AutoModel checkpoint.

Inspect the bundle, then explicitly upload to a **new private model repository**:

```bash
.venv-visual/bin/python -m safetyjev.model_export upload \
  --folder outputs/visual-training/reference-export --repo-id '<owner>/<model-repo>'
```

This does not overwrite an existing repository. If upload fails after repository
creation, inspect that destination before retrying with the standard HF CLI; do
not delete it or make it public automatically. Record the returned HF revision.

Return the code/submodule/data revisions, actual configuration, validation and
test reports, W&B link (if enabled), HF model revision, and throughput/memory
measurements to the project owner. Preserve the local run until those artifacts
have been received. Specialized inference optimization and online interventions
come after this classifier experiment.

## Inference

```bash
HF_HOME="$PWD/checkpoints/hf_cache" .venv-visual/bin/python -m safetyjev.visual_predict \
  --checkpoint outputs/visual-training/reference-run/final/model \
  --overview /path/to/current-overview.png --wrist /path/to/current-wrist.png \
  --question 'Is the jar off its supporting surface with its lid open?'
```

`score_images` also accepts a mapping of named questions and returns the familiar
`answers.<name>.noul` probability for each question. It uses the reference native
multimodal forward; this is not an accelerated HTTP service or a future-action
predictor.

Offline package preparation accepts `--media-workers N` (1–64, default 1)
for bounded image validation processes. Every referenced image is still fully
decoded and hashed; parallel execution preserves index order and labels. A
classifier build over the same immutable sources can use `--reuse-media
<completed-judge-package>` together with `--reuse-annotations
<completed-judge-package>/annotations`. Media reuse verifies the complete source
HDF5 SHA256, the image-manifest SHA256 and reference coverage before skipping
repeated decoding. Older packages without source content hashes cannot be reused
this way. Keep the total process budget within the source host's available CPU
and memory; application callers using multiple workers need a guarded Python
entrypoint (`if __name__ == '__main__':`).
