# Current-camera visual Open-Jev training

This reference trains a current-state classifier from one overview image, one
wrist image and a natural-language question. It retains Qwen's native visual
encoder and multimodal fusion, freezes the visual and base language weights,
and trains language LoRA plus a shared scalar head. The logits are `[0, score]`
in `[No, Yes]` order. It does not generate answer text.

The text-only Open-Jev path remains available. Visual checkpoints are explicitly
identified and rejected by the text-only loader rather than silently losing
their image input. The specialized inference kernels are not used by this reference
visual forward. It is also separate from the action-conditioned runtime guard.

## Environment

Create a Python 3.11 environment and install the training dependencies:

```bash
python3.11 -m venv .venv-visual
.venv-visual/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-visual/bin/python -m pip install -r requirements-visual.txt
.venv-visual/bin/python -m pip install --no-deps -e third_party/Open-Jev -e .
```

Choose CUDA wheels compatible with the deployment host. Model weights and run artifacts live in ignored
`checkpoints/` and `outputs/` directories.

## Data and training

Prepare the H1 package following [data preparation](data-preparation.md). The
loader provides current RGB images, questions and Noul targets; source model,
ID/OOD, AP truth and episode metadata never become prompt features. The direct-video path verifies the raw source files before each run. The cached
path verifies all raw hashes during preparation, then checks package, split and
raw-manifest identities on startup and each cached image's checksum on access.
An explicit full-cache check is available below. Keep the extracted source data
alongside the cache; this changes storage access, not the dataset or its labels.

```bash
bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-smoke --device cuda:0
```

The smoke configuration uses pinned Qwen3.5-0.8B weights, four optimizer
updates and capped evaluation to verify the training pipeline. The cap reads
the first batches only; choose representative validation sampling or full
evaluation when designing the training experiment.
`five_family_visual_27b_reference.json` uses pinned Qwen3.8-27B weights. Its
1,000-update budget is a starting diagnostic budget, not a claim of convergence.
At global batch 128 this draws 128,000 pairs; the delivered train split contains
2,440,945 pairs. Balanced replacement sampling does not guarantee epoch coverage.

Training uses one device or one-node DDP, gradient accumulation, bfloat16 or float32,
language-only LoRA, gradient checkpointing, AdamW, warmup/cosine learning rates,
gradient clipping, and the upstream cross-entropy plus Brier objective. The
sampler reconstructs each epoch and batch cursor deterministically.

The reference uses train, validation and test. During training, validation NLL
on a fixed 4,096-row subset selects the checkpoint. The subset is drawn uniformly
without replacement, independently of model results, and recorded in
`validation-subset.json`; small-query coverage is not guaranteed. The 27B reference sets `evaluation.run_test=false`: finalization runs full
validation and saves the selected model without constructing a test loader.
Run complete test evaluation explicitly after selecting the experiment, as
shown in [classifier handoff](classifier-handoff.md). Older configurations that
omit this option retain automatic final test evaluation. The smoke config
instead caps evaluation batches and must not be used for quality claims.

Temperature stays at 1. Reports include per-question and pooled confusion counts,
No/Yes support, precision, recall, F1, two-class macro F1 and balanced accuracy,
as well as NLL and Brier. Metrics are calculated from individual predictions,
not unweighted averages of rank metrics. A missing class has null recall/F1;
balanced accuracy is null unless both classes exist. Yes is question-dependent,
not universally synonymous with unsafe. The current data come from reviewed
unsafe episodes: their safe frames do not measure false alarms across a natural
population of wholly safe episodes. Use per-query results when assessing rare
violations, rather than pooled accuracy alone.

## Prepare efficient input storage

After downloading and extracting the HF ZIP as described in the repository
README, run from the repository root:

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

## Choose GPU count and batch size

Run host checks, then launch the same training entrypoint:

```bash
.venv-visual/bin/python tools/visual_training_preflight.py \
  --config configs/training/five_family_visual_27b_reference.json --gpus 8

NPROC_PER_NODE=8 bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_27b_reference.json \
  --output outputs/visual-training/five-family-27b
```

Only four scaling controls are needed: `NPROC_PER_NODE`, `--batch-size`
(per-device microbatch), `--global-batch-size`, and `--workers` (per rank).
Accumulation is derived exactly, refusing non-divisible combinations:

| GPUs | Microbatch per GPU | Accumulation | Global batch |
|---|---:|---:|---:|
| 8 | 8 | 2 | 128 |
| 4 | 8 | 4 | 128 |
| 1 | 8 | 16 | 128 |

Both 27B reference configurations use per-device batch 8, global batch 128,
and two loader workers per rank. The eight-rank launch therefore uses two
accumulation steps; the worker processes total 16. Accumulation is computed at
launch rather than fixed in the configuration. Per-device batch 8 was selected
from four-rank throughput/memory tests; eight-rank execution still needs a host
check. Override the four controls above to fit another machine.

Increase microbatch only after measuring memory. A single high-memory GPU uses
`NPROC_PER_NODE=1`; no source changes are required. DDP replicates the whole
frozen backbone on each GPU, so more cards do not solve per-card model fit.
A B300 requires a compatible Torch/CUDA build; the environment above is not a
B300 validation. Preflight executes BF16 CUDA math and reports memory, architecture
and batch accounting; it does not prove full-model fit or NCCL health.

Training ranks partition one deterministic weighted draw stream. The final epoch
batch is filled to a complete global microbatch (replacement draws, or repeated
prefix indices for uniform sampling). Validation/test have disjoint partitions
with no padding, so every selected row is counted once. Workers use spawn,
worker-local handles, persistent processes and bounded prefetch. A failed rank
fails the run; it must not be interpreted as a partial successful evaluation.

## Checkpoints and resume

```text
run/
  run.json
  training_identity.json
  training.jsonl
  training_status.json
  checkpoints/step-00000002/
    model/                       # base revision + language adapter + scalar head
    training_state.pt            # optimizer, trainable weights, RNG and data cursor
    checkpoint.json              # identity and artifact hashes
  evaluations/                   # development validation results
  final/
    model/                       # selected checkpoint
    test.jsonl                   # only when evaluation.run_test=true
    report.json
```

Snapshots are written at optimizer-step boundaries. Resume restores the optimizer,
Python/NumPy/Torch/CUDA RNG and epoch/batch position. Model, data, source-code and
training-configuration identities and GPU world size must agree. Distributed
checkpoints preserve each rank's RNG; rank zero owns shared checkpoint files. The base weights are reconstructed
from their pinned revision; the snapshot does not duplicate frozen base weights.
Deterministic Torch algorithms and a deterministic cuBLAS workspace are enabled
in the reference configurations. These settings and core dependency versions are
part of the resume identity. Same-seed BF16 CUDA training without these settings
can diverge before any interruption; cross-hardware bitwise equality is not
assumed even with deterministic execution.

```bash
# Optional controlled interruption to exercise resume:
bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-resume --stop-after-step 2

bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-resume \
  --resume outputs/visual-training/five-family-resume/checkpoints/step-00000002
```

Resume uses the latest retained checkpoint in the original run directory.
Rewinding an older checkpoint in place is refused before log mutation. Trailing
logs from an interrupted update are preserved under unique backup names.
The final model and reports are built in a temporary directory and atomically
published as `final/`, so interrupted finalization can be retried from the latest
training checkpoint. Only load trusted, locally produced training snapshots;
inference model loading uses the restricted weights-only loader for its head.

## Inference

```bash
HF_HOME="$PWD/checkpoints/hf_cache" .venv-visual/bin/python -m safetyjev.visual_predict \
  --checkpoint outputs/visual-training/five-family-smoke/final/model \
  --overview /path/to/current-overview.png --wrist /path/to/current-wrist.png \
  --question 'Is the jar off its supporting surface with its lid open?'
```

`score_images` also accepts a mapping of named questions and returns the familiar
`answers.<name>.noul` probability for each question. It uses the reference native
multimodal forward; this is not an accelerated HTTP service or a future-action
predictor.

## Verification

```bash
.venv-visual/bin/python -m unittest discover -s tests -v
.venv-visual/bin/python -m unittest discover -s third_party/Open-Jev/tests -p 'test_visual_*.py' -v
```

Use the four-update smoke configuration above to check the model training path.
Engineering checks cover local small-model tests, CPU/Gloo accounting/resume,
and actual 27B four-rank training, save/reload, and throughput tests for both
models. The classifier used the full five-family cache; Predictor Judge tests
used pilot captures. Eight-rank execution and the full Predictor Judge dataset
still require their own acceptance checks. These checks establish functionality,
not trained-model quality.


## Evaluate a saved classifier independently

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_model.sh --task classifier \
  --checkpoint outputs/visual-training/five-family-27b/final/model \
  --package datasets/packages/five_family --frame-cache datasets/cache/five_family \
  --split test --workers 2 --output outputs/classifier-test.json
```

This evaluates the complete selected split exactly once across ranks and writes
per-pair predictions alongside the report. Use one GPU by setting `NPROC_PER_NODE=1`.
The same launcher accepts `--task predictor_judge`; that model has different inputs
and supervision described in [its training guide](predictor-judge-training.md).
The two trainers share batch scaling, prepared CPU inputs, frozen vision/language
LoRA, optimizer, exact-cursor resume and rank-safe finalization. Each has its own
cache/loader to preserve its dataset contract. Model-quality and target-hardware
acceptance remain separate from a successful small-model functional smoke.

## Experiment tracking and model delivery

[Classifier handoff](classifier-handoff.md) describes optional W&B logging,
validation-only development, complete standalone test evaluation, and an
inference-only HF model export. Tracking is disabled by default and does not
change the dataset, sampler, model inputs or training objective.
