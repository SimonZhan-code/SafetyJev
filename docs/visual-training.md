# Current-camera visual Open-Jev training

This reference trains a current-state classifier from one overview image, one
wrist image and a natural-language question. It retains Qwen's native visual
encoder and multimodal fusion, freezes the visual and base language weights,
and trains language LoRA plus a shared scalar head. The logits are `[0, score]`
in `[No, Yes]` order. It does not generate answer text.

The text-only Open-Jev path remains available. Visual checkpoints are explicitly
identified and rejected by the text-only loader rather than silently losing
their image input. Yiqi's specialized fast kernels are not used by this reference
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
ID/OOD, AP truth and episode metadata never become prompt features. The trainer
verifies split files, registered raw manifests and source-file hashes before
both a new run and a resume.

```bash
bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-smoke --device cuda:0
```

The smoke configuration uses pinned Qwen3.5-0.8B weights, four optimizer
updates and capped evaluation to verify the training pipeline. The cap reads
the first batches only; choose representative validation sampling or full
evaluation when designing the training experiment.
`five_family_visual_27b_reference.json` uses pinned Qwen3.8-27B weights and full held-out
evaluation; choose hardware with sufficient memory for that model.

Training uses one device, gradient accumulation, bfloat16 or float32,
language-only LoRA, gradient checkpointing, AdamW, warmup/cosine learning rates,
gradient clipping, and the upstream cross-entropy plus Brier objective. The
sampler reconstructs each epoch and batch cursor deterministically.

The reference uses train, validation and test. Validation NLL selects the
checkpoint; test data are used only for the final report. Temperature stays at 1. Reports contain
per-question confusion counts, precision/recall for Yes, NLL and Brier. Yes is
question-dependent, not universally synonymous with unsafe. A question with no
positive targets has undefined Yes recall, recorded as null.

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
    test.jsonl
    report.json
```

Snapshots are written at optimizer-step boundaries. Resume restores the optimizer,
Python/NumPy/Torch/CUDA RNG and epoch/batch position. Model, data, source-code and
training-configuration identities must agree. The base weights are reconstructed
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
