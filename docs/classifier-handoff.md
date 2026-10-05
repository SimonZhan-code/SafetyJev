# Classifier training handoff

This task trains the current-camera classifier: overview RGB, wrist RGB and a
natural-language AP question → No/Yes. The base model and vision encoder stay
frozen; language LoRA and a shared scalar head are trained. Predictor Judge data
collection and the agentic runtime loop are separate workflows.

## Fixed inputs

Use a recorded SafetyJev commit and its exact Open-Jev submodule commit. Run
`git submodule update --init --recursive` after checkout. Do not update the
submodule independently during an experiment.

Download the five-family dataset using the pinned command in the root README,
and unzip **from the repository root**. It contains 4,816 episodes and 3,113,929
image/question pairs (about 25.3 GB extracted): Jar, Lid, Stack, Cabinet and Dusty.
Keep the supplied base-task-group split for the first reference experiment:

| Split | Episodes | Pairs |
|---|---:|---:|
| Train | 3,781 | 2,440,945 |
| Validation | 534 | 334,276 |
| Test | 501 | 338,708 |

Changes to splits, question wording, sampling, model or training budget define
another experiment and must be recorded. Keep all conditions, policies and
seeds of one base task together. Do not rebalance validation or test.

## Prepare and verify the host

1. Install the pinned visual environment in [visual training](visual-training.md).
2. Extract the data, check available disk space, and prepare/verify its indexed
   frame cache on local fast storage using the commands in that guide. The cache
   requires additional space beyond the ZIP and extracted data. It is built once
   and reused across runs.
3. Download the pinned model revision before launching DDP workers.
4. Run the loader profile and GPU preflight from the same guide.
5. Test the actual target model on all intended GPUs before the experiment.
   A small-model test alone does not validate full-model memory or NCCL.

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
