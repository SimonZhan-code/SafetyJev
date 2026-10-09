# SafetyJev

SafetyJev trains constraint-conditioned visual safety judges for robotic
manipulation and provides an agentic runtime for studying how safety judgments
can guide policy execution. It includes two model pipelines: a **current-state
Classifier** and an **action-conditioned Predictor Judge**.

## Model pipelines

Both models retain the native visual encoder and multimodal backbone of the
Open-Jev fork. Natural-language questions share one scalar decision head, with
outputs in `[No, Yes]` order; there is no separate head per task family.

The current semantic pipeline uses the following contracts. The guides also
retain the separately versioned legacy AP/monitor workflows and their published
datasets; those packages are not the new semantic release.

| | Classifier | Predictor Judge (`chunk_start_v2`) |
|---|---|---|
| Question | Is the queried semantic violation present now? | Will the queried violation occur during the next eight committed actions? |
| Visual input | Current overview and wrist images | Post-action images from the previous execution segment, up to eight steps per camera |
| Additional input | Natural-language safety query | Historical executed actions and aligned robot states, current state, eight future commands, query and masks |
| Meaning of Yes | Current state violation | Future-segment violation, including one that is already ongoing |
| Supervision | Same-step semantic evidence; interval-only liquid labels excluded | Observed future semantic evidence over the committed segment |
| Trainable components | Language LoRA and shared decision head | Language LoRA, shared head and state/action projections |
| Detailed workflow | [Classifier guide](docs/classifier-training.md) | [Predictor Judge guide](docs/predictor-judge-training.md#chunk-boundary-semantic-judge) |
| Configuration | [Classifier reference](configs/training/five_family_visual_27b_reference.json) | [Qwen3.8-27B](configs/training/predictor_judge_chunk_27b.json), then [Qwen3.5-9B](configs/training/predictor_judge_chunk_9b.json) |

Raw GT is preserved. Semantic evidence is used for supervision and never fed to
the model. A historical command is paired with its post-action images and robot
state; missing history is masked. Predictor Judge runs before the first action of
each new chunk and does not generate future actions, states or images.

Both pipelines support single-device or single-node DDP training, worker-side
preprocessing, indexed caches, checkpoint resume, W&B tracking and HF model
export. Validation selects checkpoints; test evaluation is separate. The semantic
judge reports accuracy, recall, precision and confusion counts, with observed
inference duration. Ordered episode shadow replay is a separate functional check.
The Unsafe600 data release has passed data/interface acceptance. Trained-model
performance and closed-loop safety require separate experiments.

## Get started

For a new training run or agent handoff, start with the
[training handoff](docs/training-handoff.md). It provides the execution order and
links to the two model guides, from host preparation through evaluation.

### Get the code

```bash
git clone --recurse-submodules --branch feat/data-preparation \
  https://github.com/SimonZhan-code/SafetyJev.git
cd SafetyJev
```

For an existing checkout, switch to `feat/data-preparation` and run
`git submodule update --init --recursive`. Record both repository revisions for
an experiment.

### Training environment

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

Both model pipelines use this environment. W&B uses the optional tracking extra
described in each model's guide. Account credentials and tracking destinations
are configured on the training host, outside repository configuration.

### Download the current training dataset

The current semantic release is
[SafetyJev-unsafe600](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/SafetyJev-unsafe600),
pinned to `598da866544f0ea218853e3b0d8fe64c2b000ab5`: 600 episodes,
480/60/60 whole-task-group train/validation/test, approximately 302.81 GB.
Allow additional disk space for model weights, training checkpoints and optional
caches. The independent `-raw` backup is not required. Do not run a source builder,
relabel, resplit or download a legacy ZIP for this release.

Use a published SafetyJev commit containing the
[current training workflow](docs/training-handoff.md) and its data-loader
implementation. Record `git rev-parse HEAD` and
`git -C third_party/Open-Jev rev-parse HEAD` after checkout/submodule initialization.
The data has passed interface acceptance; first-run GPU/DDP and optimizer/resume
acceptance must still be performed on the training host. Code publication and
formal model training are separate from data publication.

After installing the environment above and obtaining gated dataset/model access,
authenticate on the training host (never put tokens in commands or documents):

```bash
.venv-visual/bin/hf auth login
.venv-visual/bin/python - <<'PYDATA'
from huggingface_hub import snapshot_download
snapshot_download('IDEAS-Lab-Northwestern/SafetyJev-unsafe600',
                  repo_type='dataset',
                  revision='598da866544f0ea218853e3b0d8fe64c2b000ab5',
                  local_dir='datasets/unsafe600')
PYDATA
```

Keep `episodes/`, `metadata/`, `classifier/` and `predictor_judge/` together.
Both loaders use the delivered JPEGs directly (`frame_cache=null`); decoded caches
are optional. Follow the current-release section of the selected guide next.
Start with Predictor Judge Qwen3.8-27B; the later Qwen3.5-9B run compares backbones,
not parameter count alone. The Classifier is a separate experiment.

### Follow the selected model's guide

| Workflow | Classifier | Predictor Judge |
|---|---|---|
| Current dataset, pilot and training | [Unsafe600 workflow](docs/classifier-training.md#unsafe600-start-to-finish) | [Unsafe600 workflow](docs/predictor-judge-training.md#unsafe600-start-to-finish) |
| Prepare cache and train | [Host preparation and training](docs/classifier-training.md#prepare-and-verify-the-host) | [Cache and training](docs/predictor-judge-training.md#cache-train-and-evaluate) |
| Configure tracking | [W&B](docs/classifier-training.md#optional-wb-logging) | [W&B](docs/predictor-judge-training.md#experiment-tracking) |
| Test and publish the model | [Test and delivery](docs/classifier-training.md#final-test-and-interpretation) | [Test and delivery](docs/predictor-judge-training.md#export-and-publish-the-selected-checkpoint) |

Run commands from the repository root. Unsafe600 is already constructed. The
[fixed-cohort builder](docs/data-preparation.md#fixed-cohort-chunk-build) is for
preparing a different source release, not for starting this training run. Legacy
published datasets have separate pinned revisions and extraction instructions in
the guides. Generate disposable caches on the training host as needed.

The new chunk configurations start with per-device batch 1 and global batch 128;
legacy reference configurations use per-device batch 8. Adjust `NPROC_PER_NODE`, `--batch-size`, `--global-batch-size` and `--workers`
to the host; gradient accumulation is derived automatically. Training budgets
and sampling differ between the two models and are described in their guides.

## Agentic runtime

The runtime supports three execution modes:

- **Shadow:** score proposed actions while preserving policy execution, then
  compare judgments with horizon-aligned monitor labels.
- **Guard and regenerate:** reject an action chunk and request another proposal
  from the same observation; bounded retry exhaustion ends the episode.
- **Planner-assisted repair:** use rejection feedback, cameras/state and bounded
  history to revise the VLA instruction through OpenRouter. Replacement actions
  must pass the same guard.

```text
Observation -> VLA action chunk -> safety judgment -> controller -> simulator
                                      |                               |
                               regenerate / repair              monitor labels
```

The existing runtime adapter uses a pinned ManiGuard integration and an Open-Jev
HTTP connector for a proprio-only baseline. Its recorded integration experiments
are separate from the two trained visual-model pipelines; training and offline
evaluation do not require running the intervention loop. The adapter instruments
the upstream runner in memory and checks source hashes. Simulation pauses during
model calls.

See the [runtime runbook](docs/runbook.md) for deployment and mode selection,
[protocol](docs/evaluation-protocol.md) for horizon and label semantics, and
[evaluation records](docs/evaluation-results.md) for tested versions, simulator
compatibility and measured results. Existing integration checks do not establish
improved closed-loop safety.

## Code and documentation map

| Area | Entry points | Documentation |
|---|---|---|
| Classifier data | `safetyjev/visual_dataset.py`, `safetyjev/visual_cache.py` | [Data format and construction](docs/data-preparation.md) |
| Classifier training | `scripts/train_visual.sh`, `safetyjev/visual_train.py` | [Training mechanics](docs/classifier-training.md) |
| Predictor Judge data | `safetyjev/predictor_judge_commands.py`, `safetyjev/predictor_judge_dataset.py`, `safetyjev/predictor_judge_cache.py` | [Input and supervision contract](docs/predictor-judge-training.md#chunk-boundary-semantic-judge) |
| Predictor Judge training | `scripts/train_predictor_judge.sh`, `safetyjev/predictor_judge_train.py` | [Training and delivery](docs/predictor-judge-training.md) |
| Model implementations | `safetyjev/chunk_judge_model.py`, `third_party/Open-Jev/jev/visual_model.py`, `third_party/Open-Jev/jev/predictor_judge_model.py` | [Open-Jev integration](docs/open-jev-integration.md) |
| Shared evaluation and export | `scripts/evaluate_model.sh`, `safetyjev/model_export.py` | Model-specific guides above |
| Agentic runtime | `safetyjev/cli.py`, `maniguard.py`, `capture.py`, `guard.py`, `planner.py`, `predictors.py` under `safetyjev/` | [Runtime runbook](docs/runbook.md) |

## Local verification

The unittest command covers training/runtime tests. The semantic data suite also
contains pytest functions; run it in a source-reader environment with pytest,
ManiGuard recording dependencies and licensed USD assets available. The two
environments have different optional dependencies. After preparing the relevant
environment:

```bash
.venv-visual/bin/python -m unittest discover -s tests -v
.venv-visual/bin/python -m safetyjev.cli --help
# In the prepared source-reader environment:
python -m pytest tests -q
```

For the pinned simulator-adapter checks, use
`python -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard`.
Set `SAFETYJEV_MANIGUARD_ROOT=/path/to/ManiGuard` when running tests to include
the instrumented runner in the CPU test harness. Actual simulator execution uses
the dependencies and source version specified in the runtime runbook.
