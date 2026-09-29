# Guard-and-regenerate loop

This first intervention uses one VLA and one safety guard. No second VLM, prompt
rewriting, recovery planner, online training, or oracle-guided action selection.

```text
current observation -> VLA proposal -> evaluate every constraint
                              ^                 |
                              |        all no_violation -> execute chunk -> observe
                              |                 |
                              +--- any violated/unknown
                                   (bounded retries)
```

## Decision contract

Run with `capture --execution-mode guard_regenerate`. The existing `shadow`
default still records unchanged-policy trajectories.

1. Ask the configured VLA for an action chunk.
2. Score the portion the runner would actually execute: the minimum of the
   returned length, configured execute horizon, and remaining episode steps.
   Inputs include the same gripper binarization and clipping as execution.
3. Check every individual task constraint. The existing Open-Jev HTTP adapter
   returns a score; score **below** `--guard-threshold` means `no_violation`, and
   score **at or above** the threshold means `violated`. The latter is a predicted
   violation, not an assertion that a physical violation has already occurred.
4. Only accept when every response is valid and `no_violation`. A timeout,
   malformed reply, mismatched forecast ID, nonfinite score/action, or missing
   result cannot authorize execution. No combined/global query overrides a
   failing individual constraint.
5. On rejection, request another chunk with the same observation, instruction,
   and episode seed. The pinned ManiGuard native openpi server advances its
   sampling RNG on successive inference calls within an episode. We do not
   reseed every retry or change the prompt. Duplicate candidate commands are logged.
6. `--max-regenerations 3` permits four total candidates per execution window:
   the initial proposal plus three retries. Exhaustion ends this **simulated**
   episode before executing the rejected actions. It remains an unsuccessful
   completed task outcome, with `termination_reason=guard_retries_exhausted`.
   A policy service exception remains an infrastructure failure (`crashed`).

The threshold default of 0.5 is a development setting, not a calibration result.
No passing prediction is a safety guarantee. The current scorer predicts a
**new violation within the candidate horizon**; it does not diagnose or undo an
already completed temporal violation. No privileged monitor state is sent to the
guard or used for selection. A future trained guard can replace the scoring
adapter while preserving this loop; binary outputs can map to 0/1 scores.

This version checks chunk starts only and rejects nonzero `--recheck-every` in
guard mode. The selected chunk executes through ManiGuard's original controller.
Simulation is paused during model calls, including regeneration. This is not a
physical robot stop/hold controller or a real-time deadline implementation.

## Run

First start the **ManiGuard fine-tuned** jar π0.5 checkpoint and a guard service
as described in the [runbook](runbook.md). Keep their identities and revisions
pinned. From the configured simulator environment:

```bash
python -m safetyjev.cli capture \
  --maniguard-root /path/to/ManiGuard \
  --output /data/safetyjev/jar-guard \
  --provenance /path/to/SafetyJev/configs/jar-provenance.json \
  --online-predictor /path/to/SafetyJev/configs/openjev-state-only.json \
  --execution-mode guard_regenerate \
  --guard-threshold 0.5 --max-regenerations 3 \
  -- \
  --config configs/eval/jar_transport_joint.yaml \
  --benchmark-root /data/maniguard-bench/jar_transport \
  --scenes task_0000/base --seed 0 --max-steps 64 --tag safetyjev-guard-smoke
```

Use `configs/jar-isaac51-provenance.json` only for the separately documented
experimental Isaac 5.1 environment. The node-specific launch script
`scripts/remote/guard-580-2b-isaac51.sh` runs a 64-action smoke with those paths.
It assumes both model services are already running. It has not yet been executed
on a GPU node. The supplied guard configuration still points to the preliminary,
text-only Open-Jev baseline, not a trained multimodal SafetyJev model.

## Logs and evaluation

Each episode retains the normal observation frames, monitor trace, provenance,
and ManiGuard outcome. Additional files:

| File | Contents |
|---|---|
| `candidate_forecasts.jsonl` | Every valid candidate × individual constraint, before scoring |
| `candidate_predictions.jsonl` | Guard scores or failures for those candidates |
| `decisions.jsonl` | Candidate decision, constraint verdicts, retries, duplicates, timings |
| `forecasts.jsonl`, `predictions.jsonl` | Selected candidates only, for horizon-aligned evaluation |
| `maniguard_result.json` | Original outcome plus `safetyjev_guard` counters and termination reason |

IDs include episode, execution step, candidate attempt, and constraint. A rejected
candidate has **no observed counterfactual ground truth**. Never label it using the
subsequently executed replacement's outcome. Selected windows can still be censored
if the rollout ends before their horizon. The existing `report` command includes
episode outcomes and refuses to pool different execution modes or guard settings.
Its forecast metrics in guard mode are conditioned on selection; use shadow
captures for predictor accuracy comparisons on unchanged-policy trajectories.

Compare baseline and guarded runs on the same planned scenes/seeds with the same
simulator, checkpoint, and initial policy RNG state. Reusing a policy server with
the same episode seed requires an explicit reset between runs: the pinned server
only reseeds when the seed changes. Report task success, raw violations,
engagement-gated violations, guard stops, retries, duplicates, and wall time.
Never exclude retry-exhausted episodes from the task-success denominator or count
non-engagement as useful safe completion.

## Validation record — 2026-09-29

CPU tests exercise acceptance, rejection then regeneration, a shorter replacement
horizon, unknown responses, invalid actions, retry exhaustion, duplicate proposals,
and oracle isolation. The pinned-runner test compiles its actual instrumented
while-loop and checks that only accepted replacement commands reach a controlled
`env.step`; guard exhaustion executes no further actions.

```bash
SAFETYJEV_MANIGUARD_ROOT=/path/to/ManiGuard \
  python3 -m unittest discover -s tests -v
python3 -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard
```

These are control-flow tests, not Isaac physics or model-performance results.
The previously authorized node `81.172.248.118:42195` refused SSH during this
implementation, so the new loop has not had a live π0.5 + guard + simulator run.
