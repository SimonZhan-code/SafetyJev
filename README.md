# SafetyJev

SafetyJev trains constraint-conditioned visual safety judges for robotic
manipulation and provides an agentic runtime for studying how safety judgments
can guide policy execution. It includes two model pipelines: a **current-state
Classifier** and an **action-conditioned Predictor Judge**.

## Model pipelines

Both models retain the native visual encoder and multimodal backbone of the
Open-Jev fork. Natural-language questions share one scalar decision head, with
outputs in `[No, Yes]` order; there is no separate head per task family.

| | Classifier | Predictor Judge |
|---|---|---|
| Question | Does the queried AP hold now? | Will the constraint be newly violated during the remaining commands? |
| Visual input | Current overview and wrist images | Current and two preceding adjacent frames from each camera |
| Additional input | Natural-language AP question | Current robot state, remaining actions with timing/masks, natural-language constraint |
| Meaning of Yes | Question-dependent: may describe a safe or unsafe state | A new violation within the valid action suffix |
| Supervision | Same-step AP truth | Monitor outcomes over the actually executed suffix |
| Trainable components | Language LoRA and shared decision head | Language LoRA, shared head and state/action projections |
| Data package | Five families, 4,816 episodes, 3,113,929 image/question pairs | Six families, 532 episodes, frozen action-conditioned window labels |
| Detailed workflow | [Classifier guide](docs/classifier-training.md) | [Predictor Judge guide](docs/predictor-judge-training.md) |
| Reference configuration | [Classifier config](configs/training/five_family_visual_27b_reference.json) | [Predictor Judge config](configs/training/predictor_judge_27b_reference.json) |

Monitor outputs serve as supervision, not model inputs. Predictor Judge evaluates
proposed commands; it does not generate future actions or images. The two data
packages and label meanings are distinct.

Both training pipelines support single-device or single-node DDP execution,
worker-side preprocessing, indexed image caches, gradient accumulation,
checkpoint resume, W&B tracking and HF model export. Reference runs select
checkpoints using validation; full test evaluation is a separate step. Evaluation
includes per-question or per-constraint results, rather than relying only on
pooled accuracy. Predictor Judge also reports event recall and safe-episode
false alarms.

## Get started

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

### Follow the selected model's guide

| Workflow | Classifier | Predictor Judge |
|---|---|---|
| Download, extract and verify data | [Dataset and splits](docs/classifier-training.md#fixed-inputs) | [Prepared dataset](docs/predictor-judge-training.md#use-a-prepared-dataset) |
| Prepare cache and train | [Host preparation and training](docs/classifier-training.md#prepare-and-verify-the-host) | [Cache and training](docs/predictor-judge-training.md#cache-train-and-evaluate) |
| Configure tracking | [W&B](docs/classifier-training.md#optional-wb-logging) | [W&B](docs/predictor-judge-training.md#experiment-tracking) |
| Test and publish the model | [Test and delivery](docs/classifier-training.md#final-test-and-interpretation) | [Test and delivery](docs/predictor-judge-training.md#export-and-publish-the-selected-checkpoint) |

Each guide pins its dataset revision and specifies the extraction layout. Run
its commands from the repository root. Prepared datasets already include window
indices and splits; rebuilding the dataset is not required to start training.
Generate the disposable cache on the training host and keep the source files
available alongside it.

The 27B reference configurations start with per-device batch 8 and global batch
128. Adjust `NPROC_PER_NODE`, `--batch-size`, `--global-batch-size` and `--workers`
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
| Predictor Judge data | `safetyjev/predictor_judge_commands.py`, `safetyjev/predictor_judge_dataset.py`, `safetyjev/predictor_judge_cache.py` | [Input and supervision contract](docs/predictor-judge-training.md#input-and-supervision-contract) |
| Predictor Judge training | `scripts/train_predictor_judge.sh`, `safetyjev/predictor_judge_train.py` | [Training and delivery](docs/predictor-judge-training.md) |
| Model implementations | `third_party/Open-Jev/jev/visual_model.py`, `third_party/Open-Jev/jev/predictor_judge_model.py` | [Open-Jev integration](docs/open-jev-integration.md) |
| Shared evaluation and export | `scripts/evaluate_model.sh`, `safetyjev/model_export.py` | Model-specific guides above |
| Agentic runtime | `safetyjev/cli.py`, `maniguard.py`, `capture.py`, `guard.py`, `planner.py`, `predictors.py` under `safetyjev/` | [Runtime runbook](docs/runbook.md) |

## Local verification

After installing the training environment:

```bash
.venv-visual/bin/python -m unittest discover -s tests -v
.venv-visual/bin/python -m safetyjev.cli --help
```

For the pinned simulator-adapter checks, use
`python -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard`.
Set `SAFETYJEV_MANIGUARD_ROOT=/path/to/ManiGuard` when running tests to include
the instrumented runner in the CPU test harness. Actual simulator execution uses
the dependencies and source version specified in the runtime runbook.
