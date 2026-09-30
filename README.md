# SafetyJev

An agentic robotics system that enhances safety through an in-the-loop,
safety-centric fine-tuned Jev model at runtime.

Train a shared visual decision model from the current overview image, wrist
image and a natural-language AP question. This branch includes the five-family
ManiGuard dataset interface and a native multimodal Open-Jev training path.

## Visual classifier: data and training

### 1. Get the code

```bash
git clone --recurse-submodules --branch feat/data-preparation \
  https://github.com/SimonZhan-code/SafetyJev.git
cd SafetyJev
```

For an existing checkout, switch to `feat/data-preparation` and run
`git submodule update --init --recursive`.

### 2. Put the data in place

Download **`safetyjev-five-family.zip`** from the shared Drive folder. From
inside the **SafetyJev repository root**, run:

```bash
unzip /path/to/safetyjev-five-family.zip
```

The ZIP already contains `datasets/`. After extraction, the layout is:

```text
SafetyJev/
  README.md
  safetyjev/
  third_party/Open-Jev/
  datasets/
    README.md                       # dataset contents, labels and loader example
    raw/                            # original videos and AP traces
    annotations/                    # trajectory GT
    definitions/                    # question definitions
    packages/five_family/
      train.jsonl
      validation.jsonl
      test.jsonl
      dataset_metadata.json
```

The data cover Jar, Lid, Stack, Cabinet and Dusty: **4,816 episodes, 9,632 videos,
3,113,929 image/question pairs**, about **25.3 GB extracted**. Read
[the data README](datasets/README.md) after extraction. The current train/val/test
split is provisional and grouped by base task; the training owner can revise it.

### 3. Install the training environment

The tested environment uses Python 3.11 and CUDA 12.6 PyTorch wheels:

```bash
python3.11 -m venv .venv-visual
.venv-visual/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-visual/bin/python -m pip install -r requirements-visual.txt
.venv-visual/bin/python -m pip install --no-deps -e third_party/Open-Jev -e .
```

### 4. Check data loading and start a training smoke run

Run both commands from the repository root:

```bash
.venv-visual/bin/python tools/check_data_pipeline.py \
  --package datasets/packages/five_family \
  --output outputs/data-check.json

bash scripts/train_visual.sh \
  --config configs/training/five_family_visual_smoke.json \
  --output outputs/visual-training/five-family-smoke --device cuda:0
```

The smoke run uses Qwen3.5-0.8B for four updates to check the complete training
path. For the training experiment, choose the model, batch size, training budget
and validation sampling in the configuration. A 27B starting configuration is
`configs/training/five_family_visual_27b_reference.json`. The reference trainer
uses one device, train/validation/test and a shared No/Yes head. There is no
separate calibration stage.

[Training details](docs/visual-training.md) cover checkpoint resume and inference.
[Data preparation](docs/data-preparation.md) documents rebuilding the dataset.

## Prediction evaluation and a first guarded loop

We separate two experiments:

1. **Prediction:** run the existing ManiGuard π0.5 policy unchanged, forecast
   constraint violations before actions execute, and compare with ManiGuard's
   physics-grounded monitor. This repository implements the initial shadow
   capture, labeling, predictor connection, and reporting pipeline.
2. **Intervention:** use those predictions to change execution and measure safe
   success, engagement, and overhead. The optional `guard_regenerate` mode checks
   every constraint before executing a chunk. A rejection triggers a new VLA
   proposal from the same observation; bounded retry exhaustion ends the simulated
   episode unsuccessfully. No second VLM or prompt rewriting is used.

Optional **System 1 planner**: add `--planner-config configs/openrouter-planner.json`
to guarded execution to send rejection feedback, current cameras/state, and bounded
history to OpenRouter. Its revised VLA instruction generates a replacement chunk
that must pass the same guard. See [OpenRouter setup](docs/runbook.md).

```text
ManiGuard observation -> π0.5 action chunk -> original controller -> simulator
          |                    |                                  |
          +---- SafetyJev -----+                                  |
               forecast                              ManiGuard LTL monitor
                  |                                               |
                  +---------- horizon-aligned comparison ----------+
```

**Status:** The full π0.5 + Open-Jev 2B + Isaac Sim 5.1 + DeepSeek V4.1 Flash
loop ran on one RTX PRO 6000 Blackwell (256 GiB disk). Five live planner calls
returned valid instructions; three repaired replacement chunks passed the guard
and executed. The planner case reached its 64-action cap, while guard-only
regeneration stopped at 24 actions after retry exhaustion. Peak total GPU memory
was 18.2 GiB; median planner latency was 2.06 s. All 55 CPU tests pass.

This validates integration, **not improved safety**. All three short cases failed
the task, never contacted the target objects, and had a raw violation at step 3.
The 0.35 guard threshold deliberately exercises rejection; the critic is still
an untuned, text-only baseline. Initial proposals also differ slightly across
same-seed runs, so this is not a controlled performance comparison.
See [evaluation results](docs/evaluation-results.md) and the [runbook](docs/runbook.md).

Earlier shadow evaluation: on an RTX PRO
6000 Blackwell with driver 580.126.09, the ManiGuard fine-tuned π0.5 jar policy,
Open-Jev, and simulation ran together using a separate **Isaac Sim 5.1
compatibility environment**. The 2B pilot completed 2,000 actions; the 4B base
head, 9B, and 27B variants each completed a 64-action coexistence check.

The original pinned Isaac Sim 4.5 stack starts on this host but produces heavily
noisy policy-camera images on Blackwell. Its rollout was aborted and excluded.
The newer simulator renders clean images, but benchmark equivalence is unverified.
The full pilot failed the task and has only two per-constraint positive events;
it does not establish safety-model accuracy. See the
[driver-580 compatibility report](docs/evaluation-results.md) for results,
limitations, exact versions, and reproducible artifacts. Earlier model-only
measurements are retained in the same consolidated results record.

The first policy is
[ManiGuard's π0.5 jar checkpoint](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora),
with pinned resource revisions in `configs/resource-manifest.json`. For this
development pilot, step `7400` is selected as the largest released training step;
we have not established that it was the paper's selected evaluation snapshot.

## Local verification

```bash
python3 -m unittest discover -s tests -v
python3 -m safetyjev.cli --help
python3 -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard
```

The core labels, metrics, HTTP adapter, and CLI use only the Python standard
library. NumPy is required for CPU action/guard tests and by the
simulator adapter. ManiGuard supplies its own simulator/image dependencies.

To also execute the instrumented upstream loop in the CPU test harness, set
`SAFETYJEV_MANIGUARD_ROOT=/path/to/ManiGuard` when running the test command.

See [the runbook](docs/runbook.md) for capture and model comparison commands and
[the evaluation protocol](docs/evaluation-protocol.md) for the target definition.

## Supported scope

- Pinned ManiGuard commit `be97624e0acbec6b6f9260a08891b04168eb8e6c`.
- Absolute joint `(H, 8)` actions, matching the released jar configuration;
  camera and robot-state snapshots captured before execution.
- Chunk-start predictions; optional remaining-action rechecks in shadow mode.
- Per-constraint and combined-task (`__all__`) bad-prefix labels.
- Offline replay, synchronous shadow prediction, or synchronous guard-and-regenerate execution.
- Open-Jev HTTP Noul connector for the explicitly named **proprio-only ablation**.

Zefan-Cai/Open-Jev's current loader discards the vision tower. The included
connector therefore sends robot states, recent robot-state history, action
commands, task instruction, and static constraint definitions, **not images or
simulator object-state ground truth**. This is an intentionally information-limited
baseline, not the intended full SafetyJev model. PNGs are retained for a future
multimodal scorer, which can emit predictions in the same JSONL contract.

## Implementation

| File | Role |
|---|---|
| `safetyjev/maniguard.py` | Hash-checked, in-memory upstream runner instrumentation |
| `safetyjev/capture.py` | Pre-action snapshots and independent per-constraint oracle replay |
| `safetyjev/guard.py` | All-constraint gating, bounded VLA regeneration, and candidate decision logs |
| `safetyjev/planner.py` | Optional OpenRouter instruction repair with current cameras and bounded history |
| `safetyjev/labels.py` | Horizon alignment, censoring, and monitor-gap handling |
| `safetyjev/predictors.py` | Explicit input allowlist and Open-Jev HTTP scoring |
| `safetyjev/metrics.py` | Confusion matrix, ranking, calibration diagnostics, coverage, timing |
| `safetyjev/cli.py` | Capture, predict, report, and integration verification |

The adapter does not modify upstream files. It refuses unknown source versions.
Action arrays are copied for prediction; gripper binarization and clipping match
the supported runner. The original goal checker and passive monitor remain in use.
Shadow mode preserves the policy actions; guard mode can replace a chunk or end an
episode after retry exhaustion. Simulation pauses during model calls; this is not
a real-time controller.
