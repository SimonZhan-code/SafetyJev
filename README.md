# SafetyJev

An agentic robotics system that enhances safety through an in-the-loop,
safety-centric fine-tuned Jev model at runtime.

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

**Current two-node plan:** 200 base and 800 OOD scenes have exclusive ownership
in `configs/two-node-assignments.json`. Node A finishes the original 174 non-Jar
base cases, then target and language OOD. Node B runs full Jar base first, then
environment and location OOD with families/scenes in reverse order. Clutter
remains VLA plus simulator only. The first complete PDF covers all 200 base cases
without waiting for OOD. The hourly chat follow-up is paused at the user's request.
Node B provisioning and final reports are still pending; see
[the queue and report recipe](docs/runbook.md#two-node-queue-and-base-first-pdf).

**Full non-Jar base sweep:** A managed sweep now covers 174 scenes with the
family-specific fine-tuned π0.5 policies from the evaluated checkpoint collection.
SafetyJev scores 119 Lid/Stack/Dusty/Cabinet scenes; the 55 Clutter scenes run
VLA plus simulator truth only, as requested. All five 64-action integration
checks passed (117 classifications, zero request failures). Full-length results
are pending; these checks must not be presented as the completed sweep.
See the [sweep record](docs/evaluation-results.md#full-non-jar-base-sweep-october-6)
and [operations recipe](docs/runbook.md#full-non-jar-base-sweep).

**Latest result (October 6):** The supplied trained 27B visual SafetyJev,
ManiGuard fine-tuned π0.5, and Isaac Sim 5.1 ran together on one RTX PRO 6000.
Three base Jar scenes produced 495 current-state classifications over 99 camera
pairs with no failed requests. Accuracy was 79.8% raw / 83.0% calibrated, but
both variants missed all 39 oracle-positive open-while-off-support frames.
Support/contact labels need further physical validation. Median HTTP latency
was 309 ms for all five questions; peak combined memory was 65.3 GiB.
All 61 CPU tests passed on the node.

This checkpoint consumes images and trained predicate questions. It does **not**
condition on proposed actions or forecast their future violations. The new
`visual_classification` observer evaluates current AP labels separately from
the future-window pipeline. These three development scenes do not establish
held-out generalization or improved closed-loop safety. See
[evaluation results](docs/evaluation-results.md) and the [runbook](docs/runbook.md).

The earlier π0.5 + untuned Open-Jev 2B + DeepSeek planner integration executed
three repaired replacement chunks on the same GPU as simulation. It validates
controller wiring; it did not improve task success. That experiment and its
limits remain documented separately.

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
- Trained native visual SafetyJev current-frame classification with two cameras,
  exact trained questions, frozen release calibration, and step-aligned AP labels.

The older text-only Open-Jev connector discards the vision tower and receives
robot state, history, action commands, instructions, and static constraints.
The new `visual_server` uses the release's pinned native visual loader, preserving
the image encoder, LoRA, and classification head. Its image/question-only contract
is separate from action-conditioned forecasting; neither connector receives
simulator AP truth or outcome labels.

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
| `safetyjev/visual_runtime.py` | Current-camera classification capture, aligned AP labels, raw/calibrated reporting |
| `safetyjev/visual_server.py` | Pinned native visual checkpoint serving and batched trained questions |

The adapter does not modify upstream files. It refuses unknown source versions.
Action arrays are copied for prediction; gripper binarization and clipping match
the supported runner. The original goal checker and passive monitor remain in use.
Shadow mode preserves the policy actions; guard mode can replace a chunk or end an
episode after retry exhaustion. Simulation pauses during model calls; this is not
a real-time controller.
