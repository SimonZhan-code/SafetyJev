# SafetyJev

An agentic robotics system that enhances safety through an in-the-loop,
safety-centric fine-tuned Jev model at runtime.

## First milestone: prediction evaluation

We separate two experiments:

1. **Prediction:** run the existing ManiGuard π0.5 policy unchanged, forecast
   constraint violations before actions execute, and compare with ManiGuard's
   physics-grounded monitor. This repository implements the initial shadow
   capture, labeling, predictor connection, and reporting pipeline.
2. **Intervention:** use those predictions to change execution and measure safe
   success, engagement, and overhead. Deferred until prediction evaluation works.

```text
ManiGuard observation -> π0.5 action chunk -> original controller -> simulator
          |                    |                                  |
          +---- SafetyJev -----+                                  |
               forecast                              ManiGuard LTL monitor
                  |                                               |
                  +---------- horizon-aligned comparison ----------+
```

**Status:** All 24 CPU evaluation tests and pinned source-hook checks pass.
The released π0.5 jar checkpoint also passed a GPU serving smoke test on a rented
RTX 4090: three synthetic observations produced finite `(16, 8)` action chunks.
Two warmed requests took 111 ms and 107 ms; these are not rollout benchmarks.
See the [server smoke-test report](docs/server-smoke-test.md) for evidence and
reproduction details. No Isaac Sim rollout, Open-Jev inference, safety fine-tuning,
or safety-accuracy measurement has run yet. Simulator installation awaits the
required license acceptance; host-driver compatibility is still unverified.

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
library. NumPy is optional for one CPU action-contract test and required by the
simulator adapter. ManiGuard supplies its own simulator/image dependencies.

See [the runbook](docs/runbook.md) for capture and model comparison commands and
[the evaluation protocol](docs/evaluation-protocol.md) for the target definition.

## Supported scope

- Pinned ManiGuard commit `be97624e0acbec6b6f9260a08891b04168eb8e6c`.
- Absolute joint `(H, 8)` actions, matching the released jar configuration;
  camera and robot-state snapshots captured before execution.
- Chunk-start predictions by default; optional rechecks of remaining actions.
- Per-constraint and combined-task (`__all__`) bad-prefix labels.
- Offline replay or synchronous online shadow prediction; no intervention.
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
| `safetyjev/labels.py` | Horizon alignment, censoring, and monitor-gap handling |
| `safetyjev/predictors.py` | Explicit input allowlist and Open-Jev HTTP scoring |
| `safetyjev/metrics.py` | Confusion matrix, ranking, calibration diagnostics, coverage, timing |
| `safetyjev/cli.py` | Capture, predict, report, and integration verification |

The adapter does not modify upstream files. It refuses unknown source versions.
Action arrays are copied for prediction; gripper binarization and clipping match
the supported runner. The original goal checker, action cadence, and passive
monitor remain in charge of the rollout. Instrumentation adds wall-clock overhead;
behavioral parity still requires a real matched-seed runtime check.
