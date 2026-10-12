# Configurable native predictor loop

The Unsafe600 step-500 predictor evaluates an eight-action proposal against the
release's applicable semantic questions. It uses executed history and proposed
controller commands. The simulator pauses during selection; this is synchronous
simulation, not a demonstrated real-time controller. The old `guard_regenerate`,
shadow, and per-step classifier modes remain separate and retain their behavior.

## Decision flow

```mermaid
flowchart TD
    A[Current observation and executed history] --> B[Clear decision candidate pool]
    B --> C[VLA proposes eight future actions]
    C --> D[Transform gripper and clip to controller bounds]
    D --> E[Native predictor: one binary answer per trained query]
    E --> F{Complete valid predictions?}
    F -->|Yes| G[Save candidate and count No answers]
    F -->|No| H[Record invalid candidate; no fallback eligibility]
    G --> I{All answers No?}
    I -->|Yes| X[Execute selected candidate's exact commands]
    I -->|No| J{Local retries left?}
    H --> J
    J -->|Yes| C
    J -->|No| K{Decision repair and episode call budgets left?}
    K -->|Yes| L[Consume call; OpenRouter revises VLA instruction]
    L --> M{Valid instruction returned?}
    M -->|Yes| N[Reset local retry budget; retain candidate pool]
    N --> C
    M -->|No| O{Any valid candidate in pool?}
    K -->|No| O
    O -->|Yes| P[Select most No answers; earliest breaks ties; mark fallback]
    P --> X
    O -->|No| Q[Infrastructure failure; execute nothing]
    X --> R[Append each actual action and its resulting images and state]
    R --> S{Episode finished?}
    S -->|No| A
    S -->|Yes| T[Save task outcome and separate ManiGuard safety outcomes]
```

A fallback can still predict violations. It is an explicitly recorded experimental
choice to continue, not a safety guarantee or an accepted all-safe prediction.
Early goal completion can interrupt an approved chunk. Its unexecuted suffix
never enters history and has no observed outcome label.

## Configuration

Copy/edit `configs/agentic-loop.example.json`; the runnable private file is
`configs/agentic-loop.local.json`. Both credential fields are empty in the example.
The local file is ignored by Git and must have mode `0600`. Paths are resolved
relative to the configuration file. Do not put this file in artifacts or backups
intended for publication. Runtime artifacts contain validated configuration fields
with `api_key` and `hf_token` removed; keys are not model inputs or log fields.

| Field | Default | Meaning |
|---|---|---|
| `local_retries` | 2 | Additional VLA calls after the first proposal of each phase; 0–8 |
| `repairs_per_decision` | 1 | Maximum instruction repairs at the current state; 0–4 |
| `system1.max_calls_per_episode` | 3 | Shared call budget across all episode decisions; 0 disables |
| `execution_horizon` | 8 | Fixed by this checkpoint, not freely adjustable |
| `constraint_ids` | `null` | All applicable trained queries; or explicit applicable query IDs |
| `acceptance` | `all_no` | Stop searching when every query predicts No violation |
| `score` | `satisfied_constraints` | Count No answers; never use confidence magnitudes |
| `fallback` / `tie_break` | `best_valid` / `earliest` | Highest count among complete candidates, earliest on a tie |
| `predictor.endpoint` | `http://127.0.0.1:8794` | Local native service; remote/state-only endpoints unsupported |
| `predictor.timeout` / `batch_size` | 120 s / 1 | Request timeout and questions per GPU forward; batch size 1–8 |
| `predictor.checkpoint_dir` | `../checkpoints/unsafe600` | Pinned HF snapshot tree, including model and checkpoint manifest |
| `predictor.hf_token` | empty in example | Private HF read credential for downloads |
| `system1.enabled` | `true` | Enable instruction repairs |
| `system1.model` | `deepseek/deepseek-v4.1-flash` | Configurable OpenRouter model ID |
| `system1.api_key` | empty in example | Private OpenRouter credential |
| `system1.include_images` / `reasoning_enabled` | `true` / `false` | Current dual-camera images and provider reasoning option |
| `system1.timeout` / `max_tokens` | 30 s / 256 | HTTP timeout and response token budget |
| `system1.history_limit` | 8 | Last execution steps and last candidates sent to System 1; 1–64 |

Decision settings are validated, including when System 1 is disabled. Unsupported
scoring rules and horizons fail validation. `--agentic-config` conflicts with
explicit legacy `--execution-mode`, `--online-predictor`, `--planner-config`,
`--guard-threshold`, `--max-regenerations`, and `--recheck-every` flags. Put all
capture flags before the `--` separating ManiGuard arguments.

### Budgets and scoring example

Defaults allow `(1 + 2) × (1 + 1) = 6` candidates per decision, provided an episode
repair call remains and the repair succeeds. A successful instruction repair
starts a new three-candidate phase. Acceptance exits immediately. A failed repair
consumes its episode call and ends search; it does not create a fresh phase.
When three episode calls have been used, later decisions still get three local
candidates but no further repairs. Budgets restart with a new episode.

For three queries, candidates `A=[No,Yes,No]`, `B=[Yes,No,No]`, and
`C=[Yes,Yes,No]` score 2, 2, and 1. On exhaustion, A executes even if B and C were
generated after a repair. `[No,No,No]` would execute immediately. Missing or
malformed answers have no score and cannot win. Duplicate proposals consume VLA
attempts; identical questions and complete predictor inputs reuse a successful
prediction within that decision. Failed predictions are not cached. The pool
and cache are discarded when a new decision begins.

## Checkpoint and input contract

- HF repository: `IDEAS-Lab-Northwestern/SafetyJev-Checkpoints`.
- Revision: `7c1c0fc46227eca60473577a4450b86f93403603`.
- Model directory: `predictor-judge-27b-unsafe600/step-00000500/model`.
- Backbone metadata revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Published SafetyJev source: `af22cbb184641e781c9f1527d6d70a45a25fcc9d`.
- Open-Jev source: `9423d41e003f072e2433835913ec945b99e1a86e`.

The vendored `ChunkJudgeModel.load()` is unchanged. It loads the saved processor,
LoRA, classifier head, state/action projections, normalization and pinned backbone.
The integration pins the semantic-definition digest. The service checks
loader/dependency source hashes, checkpoint manifest file
hashes and model dimensions before readiness. Health records the checkpoint
manifest digest and model identity; the client checks that identity on health
and every inference reply. There is no fallback to the old state-only adapter.

The published chunk loader SHA256 is
`013c4c1c62a657472ddf680d7db7993df511351a333b8d6e6fc66f7b9707ce1f`.
Training metadata instead records
`d33dca3d21c7cf6a73432536fc71e7759f7fab0a60557b7c40a446ec5b7ee39d`
for the unpublished AMD training patch. They are different. NVIDIA is the supported
inference target; AMD is rejected. Published-loader CPU tests do not establish
numerical parity with that training run.

At decision time t the service receives:

- Two cameras, each with eight history image slots.
- Eight executed commands paired with their **resulting** observations and robot
  states: action a(t−8) with o(t−7), through a(t−1) with o(t).
- Current 16D state: seven arm joint positions, seven joint velocities, mean
  gripper joint position, mean gripper velocity. Read directly from the robot;
  the VLA's `obs["states"]` is not used for predictor state.
- Eight proposed 8D absolute-joint controller commands, action interval, and masks.
  Gripper binarization and controller clipping happen before prediction; the
  chosen transformed commands are exactly those passed to the runner.

At initialization/reset, only the final image slot contains the initial view;
all history-action masks are false and historical numeric slots are zero.
After a complete executed chunk all eight action-result slots are present.
Rejected proposals never change history. Saved native image preprocessing handles
RGB images without resizing/truncating history in the integration. External
constraint context is fixed to `{ "source": "none", "text": "" }`.

Configure ManiGuard with `execute_horizon: 8`, `gripper_binarize: true`, joint state
mode, eight action dimensions, and no IK conversion. The episode action cap must
be positive and divisible by eight. A policy may return a longer horizon; only
its first eight commands are proposed for execution. Short future chunks are
invalid and never padded. The runtime verifies the trained absolute JointController
and binary gripper configuration.

### Questions and label meanings

`configs/predictor-semantic-definitions.json` is the published definition set.
Binding uses the release's subject wording and static scene object scopes; it
does not inspect ground-truth AP values to choose questions. Query IDs are
`semantic_id:source_ap`, except the two conjunction questions, whose IDs are their
semantic IDs. Unsupported AP coverage, query IDs, or empty object scopes fail.
This is trained semantic coverage, not a general evaluator for arbitrary LTL.

| Trained semantic target | Difference to retain when interpreting ManiGuard outcomes |
|---|---|
| Excessive tilt | Release threshold 15 degrees; do not assume native upright predicates use this threshold |
| Falling off work surface | Release geometry uses the support plane and a 0.02 m margin; not automatically equal to native dropped APs |
| Liquid spill | Release uses 20% net contained-particle loss and a nonzero initial count |
| Direct food contact | Release contact evidence; keep it separate from task success and engagement gating |
| Open jar off support | `not jar_on_support AND not jar_closed` within the prediction window |
| Uncovered container off support | `not container_on_support AND not lid_on_container` within the prediction window |

Labels include violations already present at the boundary when they occur in the
queried window; a No answer is not a claim that earlier episode violations never
happened. The model's positive-logit difference ≥ 0 implements the release's
probability ≥ 0.5 Yes rule. Neither ranking nor System 1 receives probabilities.
ManiGuard AP/LTL traces remain separate observed outcomes. This implementation
does not reconstruct the release's physical semantic ground truth online; do not
report native prediction accuracy by silently relabeling ManiGuard APs.

## System 1, records and failures

The existing OpenRouter adapter posts to
`https://openrouter.ai/api/v1/chat/completions`, authenticates with a Bearer header,
and requests strict JSON containing only `vla_instruction`. It supplies current
images, current state, proposed commands, binary constraint feedback, and bounded
candidate/execution history. It validates response shape, finish reason, refusal
and instruction length. Invalid replies and HTTP failures consume the call budget
and trigger best-valid fallback. The instruction applies to the replacement
chunk search; subsequent decisions start with the original task.

Per-episode records:

- `episode.json`: redacted config, native provenance, semantic questions and scopes.
- `native-observations.jsonl` and PNGs: actual executed history and 16D states.
- `candidate-inputs.jsonl`: numeric inputs, questions, image references and input digest.
- `candidate-decisions.jsonl`: binary labels, count, validity, repair/local index,
  duplicate flag and timing. Initial VLA generation occurs in ManiGuard before
  selection, so its recorded generation latency is null;
  regeneration latency is measured. Prediction latency includes input assembly/cache lookup.
- `planner.jsonl`: successful repairs and sanitized failure types.
- `selections.jsonl`: selected exact commands/instruction, timing and explicit fallback flag.
- `executed-candidates.jsonl`: only executed candidates, linked to observed oracle
  steps and ManiGuard violations. Rejected proposals have no observed outcome.
- Existing `oracle.jsonl`, episode completion and simulator results remain available.

All-invalid candidates raise `AgenticInfrastructureError` and produce a crashed
infrastructure result, not a normal unsuccessful task. VLA generation/service
failure and invalid execution alignment also stop the rollout as infrastructure
errors. Candidate inference failures instead permit bounded local retry and
selection from other valid candidates. Startup/configuration failures stop before
any proposal executes; if they occur before episode construction completes there
may be no complete episode record. Credentials and response bodies are not logged
in service/planner errors.

## Launch and verification

Use separate NVIDIA model and simulator environments as in the existing runbook.
Install this repository editable so its release definition file is available:

```bash
pip install -e '.[agentic]'
# In the model environment, install the pinned native Open-Jev implementation:
pip install 'open-jev[train] @ git+https://github.com/666harrypeng/Open-Jev.git@9423d41e003f072e2433835913ec945b99e1a86e'
chmod 600 configs/agentic-loop.local.json
python -m safetyjev.agentic_server --config configs/agentic-loop.local.json --download
```

Keep the native service running. In the simulator environment, start the existing
family-specific fine-tuned pi0.5 policy service and use the corresponding ManiGuard
evaluation YAML and pinned policy provenance. Example (paths are deployment-specific):

```bash
python -m safetyjev.cli capture \
  --maniguard-root /workspace/ManiGuard \
  --output artifacts/native-jar-smoke \
  --provenance configs/jar-isaac51-provenance.json \
  --agentic-config configs/agentic-loop.local.json \
  -- --config /workspace/ManiGuard/configs/eval/jar_transport_joint.yaml --max-steps 256
```

Confirm the YAML selects horizon 8, joint mode and the intended trained policy.
A matching `/health` response is necessary but does not establish simulator parity.
The prior Isaac 5.1 compatibility caveat remains in the runbook.

Stop the predictor server before running the replay checker, which loads its own
model. With renewed HF access, it downloads the pinned original sample media and
compares NVIDIA outputs to `test-sample/predictions-step500.jsonl`:

```bash
python -m safetyjev.agentic_replay \
  --config configs/agentic-loop.local.json --download \
  --output artifacts/native-replay.json
# Add --all for every published sample, or --sample-root for existing sample files.
```

Default smoke selection covers the first reset and full-history sample for each
available family. The JSON reports probability and logit differences, binary
disagreements, latency and source digests. Default probability tolerance is 0.02
(`--probability-atol`); any disagreement or exceeded tolerance exits nonzero after
saving the report. This tolerance is a diagnostic, not evidence of statistical
or safety equivalence. Probabilities here are for numerical comparison only.

CPU verification:

```bash
SAFETYJEV_MANIGUARD_ROOT=/path/to/pinned/ManiGuard \
MANIGUARD_SAFETY_MONITOR_SOURCE=/path/to/verified/clutter/safety_monitor.py \
python -m unittest discover -s tests -v
```

Tests cover decision budgets, earliest ties, cross-repair fallback, malformed and
partial predictions, repair failures, credential redaction, native history/reset
alignment, controller transforms, replay input assembly and the pinned runner
with controlled model/simulator boundaries. These tests do not load the 27B
checkpoint. The October 11 NVIDIA smoke executed 256 actions on one base Jar scene, including
three live DeepSeek repairs and seven fallback selections. All 102 CPU tests also
passed on that node. The GPU replay comparison did **not** pass: 1 of 12 binary
answers disagreed with the saved release outputs. The task was unfinished at the
smoke limit and the native monitor recorded a raw violation. This validates
integration paths, not safety improvement or numerical equivalence. Renewed HF
access worked for the pinned downloads. See the [measured results](evaluation-results.md#native-predictor-agentic-smoke-october-11).
No training or new benchmark sweep is included.

To reproduce the bounded single-task audit after starting both model services:

```bash
PYTHONPATH=/workspace/SafetyJev /workspace/native-env/bin/python \
  scripts/remote/native-agentic-smoke.py \
  --config configs/agentic-loop.local.json \
  --output artifacts/native-jar-new-run --max-steps 256
```

Use a fresh output path. The helper records GPU memory, native health, candidate
and execution counts, exact command/instruction alignment, and the separate task
outcome. It requires the documented `/workspace` layout. The tested simulator
needed OmniGibson's `[eval]` extra and `pyarrow==21.0.0` alongside NumPy 1.26.0;
newer PyArrow 26 failed to import with that simulator's NumPy pin. Explicit
`torch==2.7.0+cu128` / `torchvision==0.22.0+cu128` in the simulator and
`torch==2.7.1+cu128` / `torchvision==0.22.1+cu128` in the policy environment prevent
an already-installed incompatible CUDA build from satisfying an unqualified pin.
The predictor used `torch==2.14.1+cu130`. Full package lists are in the saved result
record. Stop model services after the test; rental shutdown is a separate action.

### Cloud Jar comparison on the prepared Vast.ai node

`scripts/remote/jar-agentic-eval.py` uses the existing `/workspace` simulator,
policy and native predictor environments. It is not a fresh-machine installer.
The full plan is 26 pinned Jar base scenes × seeds 0/1/2 × baseline, local retries,
and System 1: 234 episodes. `--pilot` selects the first five sorted scenes and
seed 0 (15 episodes); use a separate output directory for this development pilot.
Every episode uses the original 2,000-action cap. Baseline records simulator truth
without a guard; local disables instruction repairs; System 1 uses the private
config defaults. The policy service restarts before each episode to reset its
sampling stream even when consecutive methods have the same episode seed.

On the already prepared node (`root@154.59.156.28`, SSH port `33632`):

```bash
cd /workspace/SafetyJev
/workspace/native-env/bin/python scripts/remote/jar-agentic-eval.py --prepare
/workspace/native-env/bin/python scripts/remote/jar-agentic-eval.py --launch
```

The supplied credentials live only in `configs/agentic-loop.local.json` (0600).
The launcher derives private method configs and writes redacted plans. A cloud
worker needs its own authorized SSH key; a new chat does not transfer local SSH
credentials. The node already contains the deployed source, so no GitHub token is
needed there. Pull the committed code into the cloud checkout before editing;
do not replace the deployed runtime during an active queue.

Supervisor job `safetyjev-jar-agentic-eval` survives chat/SSH disconnection. Read
`artifacts/jar-agentic-id/progress.json`, `summary.json`, `queue.log`, and per-case
capture logs. `--check` performs a CPU-only setup check. The queue refuses other
active SafetyJev jobs or GPU compute processes, checks source hashes on resume
and between cases, reserves 20 GB of disk, and stops at the first infrastructure
failure. Completed cases are audited and skipped on resume. Interrupted captures
with complete valid records are recovered; otherwise they need inspection and
explicit `--retry-failed`, which archives the prior attempt. Stop a running queue
with `supervisorctl stop safetyjev-jar-agentic-eval`; restarting uses the same
output and skips completed cases. Model services stop on exit; rental shutdown
is deliberately separate. Back up raw outputs off-node before deleting a rental.

Rates in the JSON summary exclude infrastructure failures and show per-seed
coverage and unweighted seed means. Partial coverage is provisional; only a full
paired grid supports the planned comparison. Raw and engagement-gated violations
remain separate. Candidate labels, repair timing, selections, images and executed
commands are retained in episode records. The known 1/12 replay disagreement,
experimental simulator parity, and unverified training-group independence remain
limitations, not resolved by this launcher. Setup/CPU tests do not constitute a
completed 234-episode evaluation.

The first full queue stopped before the local method executed an action: cold JAX
compilation exceeded the policy client's WebSocket pong deadline. The launcher
now runs `warm-jar-policy.py` before capture with a bounded 300-second transport
window. Its dummy inference uses seed 2^31, outside ManiGuard's 31-bit episode
seed range; the real episode therefore resets the sampler. No warm-up actions
enter the simulator or predictor history. This applies to every method. Existing
completed episodes remain valid; keep the prior source plan when migrating a
stopped queue to this launcher fix. Warm-up time is saved separately; total case
wall time includes warm-up. This changes startup handling, not task horizons,
candidate budgets, model weights or safety semantics.

To evaluate only the two predictor-guided methods, pass
`--methods local system1` to both preparation/check and launch commands. This
selects 156 episodes (26 scenes × three seeds × two methods). Existing baseline
captures may remain on disk; they are excluded from active coverage and summary
rates, with an explicit historical-record count. On October 12 the user removed
baseline reruns from the active queue. Its previous plan and baseline records
were preserved; an in-flight predictor startup caught in the stop transition
was archived and retried. Published ManiGuard
baseline results are external references, not new paired control episodes under
this simulator deployment. Changing the selected methods on an existing queue
requires stopping it and explicitly archiving/migrating the plan first.

### Constraint batching transition

`predictor.batch_size=3` groups the three Jar semantic questions for **one**
candidate in a single model forward. It does not batch candidate generation or
change acceptance, retry budgets, repair budgets, input history, or horizons.

`benchmark-constraint-batch.py` reconstructs exact saved requests (including image
bytes), verifies their original request hashes, and compares batch sizes 1 and 3
on 20 candidates spanning five completed episodes. It alternates execution order,
warms both batch shapes, repeats each twice, and records latency, peak CUDA memory,
logit differences, and binary disagreements. This is inference replay, not another
simulator evaluation. The report separates its model-only measurements from HTTP
checks and the later live rollout timings.

For the October 12 change, `apply-constraint-batch.py` runs as the one-time
Supervisor job `jar-constraint-batch-transition`. The queue dispatcher is suspended
while its current simulator child finishes. The helper preserves that completed
case, shuts down the old services, runs replay with pi0.5 resident, and checks
additional simulator memory headroom. It enables batch 3 only if sampled binary
answers agree and mean inference latency improves, followed by HTTP replay checks.
Otherwise it resumes batch 1 and records the failed validation. No completed
simulation episodes are repeated. This helper expects the explicitly saved
`batch-transition-pause.json` boundary record; it is not a general pause command.

The new public config, previous plan, migration record, and pre-switch completed
case list preserve the phase boundary. Results before/after the change must retain
their batch size; zero disagreement in these samples does not prove equivalence
on all inputs. Validation and transition status are under
`artifacts/constraint-batch-validation-20261012/`. The helper does not schedule
recurring checks, alter the checkpoint, or stop the rental.

### October 12 batching measurements

GPU replay finished on the RTX PRO 6000 with pi0.5 resident: 20 saved candidates,
three constraints each, two repeats per batch size. Full-history requests averaged
2.176 s at batch 1 versus 2.161 s at batch 3 (1.007×, 0.73% less time). Reset
requests averaged 0.969 s versus 0.466 s (2.08×). The aggregate 1.079× speedup
oversamples reset requests (five of twenty candidates), so it should not be used
as an episode speedup estimate. These timings include input decoding but exclude
HTTP, simulator execution, policy sampling and System 1 calls.

Three of 60 distinct constraint answers changed between batch sizes, consistently
on both repeats; the raw report counts six repeated disagreements. One candidate
changed from all constraints passing to rejection. Within each batch size, repeated
binary answers agreed. Peak predictor allocation increased from 49.25 to 50.45 GiB.
The cause of batch-dependent outputs remains unresolved.

The validation gate therefore rejected batch 3 and resumed the existing queue at
batch 1 after preserving 36 completed System 1 episodes. No batched simulator trial
was run, and no end-to-end speedup is established. Raw measurements and derived
grouped timings are saved in `artifacts/constraint-batch-validation-20261012/`
(`benchmark.json`, `transition.json`, and `timing-analysis.json`). Do not force batch
3 into the evaluation until its prediction differences are understood and validated.
