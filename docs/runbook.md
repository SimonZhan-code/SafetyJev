# SafetyJev runbook

This guide covers shadow capture, guard-and-regenerate execution, and OpenRouter
instruction repair. Shadow mode remains the default.

All simulator commands below are for a Linux/NVIDIA machine with ManiGuard's
BEHAVIOR/OmniGibson assets, valid Spot+Buddy, and the long-finger Franka asset.
The Mac workspace can run the CPU evaluation tests. No remote node is assumed.
Follow the [upstream setup](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
and [evaluation instructions](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/run_benchmark/).

Use separate environments for openpi policy serving, the behavior simulator, and
Open-Jev. Install this repository into the behavior environment with `pip install
-e /path/to/SafetyJev`, or put its root on PYTHONPATH. Paths below are deployment
examples; substitute the actual directories.

## 1. Pin code and resources

Use a separate ManiGuard checkout at commit
`be97624e0acbec6b6f9260a08891b04168eb8e6c`. The adapter checks hashes of the runner
and monitor implementation. It instruments a copy in memory, not files on disk.

```bash
python -m safetyjev.cli verify-integration --maniguard-root /path/to/ManiGuard
hf download IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora \
  --revision 1d84eda070313a202a595e449fcf41a1a1e8a546 \
  --include '7400/*' --local-dir /data/checkpoints/pi05-jar
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
  --revision 2ea32a1451669fb736ae78ffce9cc82aad4cceac \
  --include 'jar_transport/*' --local-dir /data/maniguard-bench
```

These downloads are large. The pinned resources were used in the recorded
driver-580 simulator experiments; see [evaluation results](evaluation-results.md).
Use at least 200 GB of disk for the documented three-environment deployment.
The NVIDIA and BEHAVIOR research-data licenses were accepted by the user for this
non-commercial evaluation.

## 2. Serve π0.5 in the openpi environment

From the configured ManiGuard checkout:

```bash
python -m maniguard.serve.openpi_native \
  --config pi05-base_datagen_v1_jar_joint_2cam_lora \
  --checkpoint /data/checkpoints/pi05-jar/7400 --port 8000
```

This released snapshot uses the native openpi/Orbax path; do not assume it is a
standalone PyTorch checkpoint. Preserve the jar configuration's left overview +
wrist cameras, absolute joint controller, assisted grasping, and execute horizon.
Record the actual loaded checkpoint and source versions. Our provenance JSON is
a declaration, not remote attestation of the policy server's loaded weights.

## 3. Capture one unchanged-policy rollout

From the ManiGuard root, in the behavior environment:

```bash
python -m safetyjev.cli capture \
  --maniguard-root /path/to/ManiGuard \
  --output /data/safetyjev/jar-smoke \
  --provenance /path/to/SafetyJev/configs/jar-provenance.json \
  -- \
  --config configs/eval/jar_transport_joint.yaml \
  --benchmark-root /data/maniguard-bench/jar_transport \
  --scenes task_0000/base --seed 0 --tag safetyjev-shadow
```

The default records predictions at chunk starts. Add `--recheck-every 1` BEFORE
the `--` separator for remaining-chunk windows after every action. This changes
the number of forecasts, not the executed policy horizon. Keep experiments separate.

Every episode gets a UUID directory:

```text
episode.json                provenance, task spec, execution config
forecasts.jsonl             one pre-action input per window × constraint
frames/                    current overview and wrist PNGs
oracle.jsonl               per-step raw monitor outcomes; evaluator-only
labels.jsonl               binary or excluded/censored window labels
maniguard_result.json       original success/safety/engagement outcome
complete.json               end marker
```

No model is needed to collect the first forecast dataset. This mode still runs
real π0.5 and ManiGuard; it is not a mock simulator. Capture failures and simulator
crashes must be inspected. The report lists incomplete captured episodes but cannot
count scene-load failures that happen before capture starts: inspect ManiGuard's
original results.jsonl and retain the complete planned-scene manifest as well.

## 4. Score the saved inputs

Serve Zefan-Cai/Open-Jev with its own loader and the pinned published
`ZefanCai/Open-Jev-2B` package revision
`0c7aa498b1627be8da4acf34c863ff0ee0a92785`. Its model card supplies the exact
upstream Qwen revision and launch instructions. The package includes an adapter
and head, not the base weights. Example once the package is downloaded:

```bash
python -m jev.server --checkpoint /data/checkpoints/open-jev-2b/package/checkpoint \
  --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8791
python -m safetyjev.cli predict --episodes /data/safetyjev/jar-smoke \
  --name openjev2b-proprio \
  --endpoint http://127.0.0.1:8791/v1/systemone \
  --model-id Qwen/Qwen3.5-2B \
  --predictor-revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 \
  --input-mode proprio_only
```

The API model ID must match the server’s `/health` identity (the upstream Qwen
name), not the Open-Jev package name or experiment label. The pinned predictor
revision above identifies the published Open-Jev adapter/head package.

For completed captures, add `--eligible-only` to offline `predict` to skip
windows excluded by the labeling rules (for example, already violated constraints).
The evaluator determines eligibility from the oracle, but sends only the original
pre-action forecast to the scorer. Selection is recorded in prediction metadata.
Use the same selection for every model; offline timing then covers the eligible
subset, whereas online capture still records all attempted requests.

This explicitly omits camera pixels. It cannot establish the quality of a
multimodal safety model. Scores from a future multimodal scorer should use the
same `forecast_id`, with JSONL rows:

```json
{"forecast_id":"episode:step:constraint", "score":0.3, "error":null, "latency_s":0.08}
```

Store `<name>.meta.json` beside them with `model_id`, `model_revision`, `input_mode`,
and `mode`. Resolve image paths relative to the episode directory. Supply only
the forecast's `input`, never `oracle.jsonl`, labels, or episode outcomes.
The illustrative score above is not a model result.

For online synchronous shadow scoring, instead add
`--online-predictor /path/to/SafetyJev/configs/openjev-state-only.json` to capture
before the separator. Each request finishes before env.step. Timeouts are recorded
as prediction failures; no prediction is used to alter actions.

## 5. Report prediction quality

```bash
python -m safetyjev.cli report --episodes /data/safetyjev/jar-smoke \
  --predictions openjev2b-proprio --threshold 0.5 --output /data/safetyjev/report.json
```

The report has a separate `global_task_forecast` block for comparison with the
combined ManiGuard monitor, plus per-constraint, episode, and family/level detail.
Use the same captures and a new prediction name for a fine-tuned checkpoint.
Existing predictions are not overwritten; mixed model revisions are rejected.

Before claiming performance: verify real source-hook execution and matched-seed
action parity, validate monitor coverage, establish held-out task groups, run both
untuned and tuned scorers on identical windows, and show positive-event counts.
No positive events means the pilot cannot estimate violation recall.

## Guard-and-regenerate loop

The default intervention uses one VLA and one safety guard. To add an OpenRouter
planner that rewrites instructions after rejection, see [System 1 setup](#system-1-openrouter-instruction-repair).
Without `--planner-config`, there is no second VLM or prompt rewriting. Neither
mode uses online training or oracle-guided action selection.

```text
current observation -> VLA proposal -> evaluate every constraint
                              ^                 |
                              |        all no_violation -> execute chunk -> observe
                              |                 |
                              +--- any violated/unknown
                                   (bounded retries)
```

### Decision contract

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

### Run

First start the **ManiGuard fine-tuned** jar π0.5 checkpoint and a guard service
as described in the [runbook](#safetyjev-runbook). Keep their identities and revisions
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
It assumes both model services are already running. The three-mode integration sweep described below exercises this
same guard path on a GPU node. The supplied guard configuration still points to the preliminary,
text-only Open-Jev baseline, not a trained multimodal SafetyJev model.

### Logs and evaluation

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

### Guard control-flow validation

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
Live API and GPU evaluation status is in [evaluation results](evaluation-results.md).

## System 1: OpenRouter instruction repair

This optional slower loop asks an OpenRouter-hosted model to repair the VLA's
instruction after a safety-guard rejection. The existing guard is still the only
mechanism that approves action execution. The planner is enabled explicitly;
plain regeneration and shadow evaluation remain available as comparison modes.

```text
observation -> VLA -> action chunk -> safety guard -> all constraints pass -> execute
                  ^                      |
                  |                  rejection
                  |                      v
                  +-- revised prompt -- OpenRouter planner
                                       ^
                      current cameras/state, constraints,
                      executed history and candidate feedback
```

### What the planner receives

- Original task, current candidate's instruction, and unchanged safety constraints.
- Current robot state, normalized remaining action commands, action convention,
  and control frequency.
- Current overview and wrist camera PNGs by default (`include_images: true`).
- Guard score and `no_violation` / `violated` / `unknown` verdict for each constraint.
- Up to eight recently executed actions with resulting observed robot states and
  the instruction that produced them.
- Up to eight recent candidate decisions, their instructions, and guard feedback.
  Rejected candidates are explicitly distinguished from executed history.

Images and this task context are sent to OpenRouter and its selected model provider.
The request uses an explicit field allowlist. It never includes ManiGuard's oracle
APs, monitor states, labels, task outcomes, or future observations. History is reset
per episode. Historical images are not included in this version; images are the
two current views. The guard's current adapter remains text-only even when the
planner uses images.

### Output and execution semantics

The planner must return a structured JSON object with only `vla_instruction`.
The client checks that it is a nonempty string of at most 512 characters, rejects
extra fields, and rejects incomplete, refused, or malformed responses. This is a
format check, not a proof that the instruction is correct or preserves the task.

The new instruction replaces `task_descriptions` in a copied observation for
the next VLA request. ManiGuard's existing openpi mapping sends it as `prompt`.
Images, robot state, and episode seed stay the same. Forecasts retain both the
original task and the actual revised policy instruction; the replacement is
checked against every original constraint before it can execute.

Instruction scope is deliberately **one replacement chunk**. A subsequent retry
can obtain another instruction; at the next observation/chunk boundary, the
initial proposal uses the original task again. There is no persistent subgoal
state or recovery-completion detector yet. The planner cannot change thresholds,
constraints, action bounds, or the task-success checker.

The existing three-regeneration limit still applies. Additionally, the default
planner budget is 20 requests per episode, with a 30-second timeout and 256-token
output cap per request. Budget exhaustion or planner failure ends the simulated
episode unsuccessfully with `planner_budget_exhausted` or `planner_error`; it
does not execute rejected actions or silently fall back to unchecked execution.
There are no automatic HTTP retries. These bounds limit requests and output,
not total dollar cost; input images and history also consume tokens.

Simulation pauses during planning and inference. This remains a synchronous
simulator experiment, not a real-time robot controller. A revised natural-language
prompt may not reliably change this checkpoint's behavior; evaluate that rather
than assuming the VLA follows the intended correction.

### Configure and run

Set `OPENROUTER_API_KEY` through the runtime machine's secret/environment setup.
Do not put the key in this repository, JSON config, CLI arguments, or chat logs.
The code reads the environment directly; it does not automatically load `.env`.
Choose an explicit model that supports image inputs and structured outputs:

```bash
export OPENROUTER_MODEL='provider/model-id'
```

That value is a placeholder; no model is silently selected. Alternatively put
the actual model ID in `model` in your planner JSON. The model ID and non-secret
settings are recorded in episode provenance. The key is used only in the HTTPS
Authorization header and is not recorded. Set `include_images: false` only for a
deliberate text-only planner ablation; unsupported vision/structured-output models
are not silently substituted.

Add the following option to the guarded capture command, **before** its `--`
separator:

```bash
--execution-mode guard_regenerate \
--planner-config /path/to/SafetyJev/configs/openrouter-planner.json
```

Keep the fine-tuned ManiGuard π0.5 policy and guard service configured as in the
[guarded-loop guide](#guard-and-regenerate-loop). The node-specific
`scripts/remote/planner-580-2b-isaac51.sh` launches a 64-action smoke in the existing
experimental Isaac 5.1 environment and requires both environment variables above.
It assumes the policy and guard services are already running.

### Artifacts and comparisons

`planner.jsonl` stores the candidate ID, bounded input context, image paths and
hashes, revised instruction, request hash, requested/returned model IDs, latency, and returned
numeric usage/cost fields when available. Images stay in `frames/`; base64 payloads
are not duplicated in logs. Failures log a sanitized error without raw server
bodies or credential-bearing headers. The system prompt is versioned in code and
hashed in provenance. `maniguard_result.json` includes planner calls, failures,
latency, and the termination reason alongside the existing guard metrics.

The report refuses to pool planner-enabled runs with plain regeneration or runs
with different planner settings. Compare the same scenes/seeds/checkpoint under:

1. Shadow/no intervention.
2. Guard plus unchanged-prompt regeneration.
3. Guard plus OpenRouter instruction repair.

Measure task success and violations together with guard stops, intervention
frequency, candidate diversity, planner calls/cost, and end-to-end delay. Preserve
the original raw versus engagement-gated safety distinction. Do not assign
counterfactual ground truth to rejected candidates.

### Planner validation

All 55 CPU tests passed, including the actual pinned ManiGuard execution loop and
its openpi observation-to-prompt mapping with controlled model/simulator boundaries.
Tests verify revised-prompt delivery, rechecking every replacement, no planner
calls on initial acceptance, bounded calls, timeout/error handling, image request
construction, schema validation, oracle isolation, and credential-free artifacts.

The implementation initially had only mocked API tests. The subsequent live
DeepSeek probe and completed three-mode simulator evaluation are recorded in
[evaluation results](evaluation-results.md).

API implementation follows the official [OpenRouter API reference](https://openrouter.ai/docs/api/reference/overview)
for authentication, chat completion requests, and image content, and
[structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs)
for JSON schema responses and `provider.require_parameters`.

## DeepSeek V4.1 Flash and recorded-observation testing

For the requested model use `configs/openrouter-deepseek-v4.1-flash.json` instead
of the generic planner config. It pins `deepseek/deepseek-v4.1-flash` and disables
reasoning explicitly with the same 256-token output cap. The generic default
remains unchanged. Incomplete responses now retain sanitized finish reason and
usage diagnostics; raw reasoning and API credentials are never logged.

For a live API-only diagnostic against a saved shadow trajectory:

```bash
PYTHONPATH=/path/to/SafetyJev python scripts/remote/planner-replay-probe.py \
  --episode /data/saved-shadow-episode --step 32 --threshold 0.35 \
  --config configs/openrouter-deepseek-v4.1-flash.json \
  --output /data/deepseek-probe.json
```

This requires `OPENROUTER_API_KEY` in the runtime environment. The selected
threshold must reject a recorded candidate. This probe uses current saved images,
recorded guard scores, and observed action/state history. It executes no VLA
request or simulator action and must not be reported as a live robotic evaluation.


## Reproduce the September 29 three-mode integration check

The tested rental node had a 256 GiB disk, 96 GiB RTX PRO 6000 Blackwell,
driver 580.159.04, and approximately 118.5 GiB container RAM. The driver advertises
CUDA 13.0; Python environments use their own CUDA-compatible wheels. Do not
replace the host driver. The simulator uses the experimental Isaac 5.1 branch
recorded below, whose equivalence to ManiGuard's original simulator is unverified.

Deploy this repository at `/workspace/SafetyJev`. Clone these source revisions:

| Directory under `/workspace` | Source | Revision |
|---|---|---|
| `ManiGuard` | `NU-IDEAS-Lab/ManiGuard` | `be97624e0acbec6b6f9260a08891b04168eb8e6c` |
| `openpi` | `Physical-Intelligence/openpi` | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `Open-Jev` | `Zefan-Cai/Open-Jev` | `3308a15ccd7eea1df7a37d6ddc39b023b801ba16` |
| `BEHAVIOR-5.1` | `StanfordVL/BEHAVIOR-1K` | `d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864` |

Create a Python 3.11 + Spot environment at `/workspace/conda/behavior51` using
conda-forge. System packages used were `libglu1-mesa libxt6 vulkan-tools libegl1
libgl1`. In that Python environment install `isaacsim[all,extscache]==5.1.0.0`
from NVIDIA's extra index, `torch==2.7.0+cu128` and `torchvision==0.22.0+cu128`
from PyTorch's cu128 index, `numpy==1.26.0`, and editable installs of
`BEHAVIOR-5.1/bddl3`, `BEHAVIOR-5.1/OmniGibson[eval]`, `ManiGuard[serve]`,
`SafetyJev`, and `openpi/packages/openpi-client`.

In `openpi`, run `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev` using Python
3.11, then upgrade its environment to `torch==2.7.1+cu128` and
`torchvision==0.22.1+cu128`. Install `ManiGuard[serve]` editable with `--no-deps`.
In a separate Python 3.11 environment at `Open-Jev/.venv`, install its `.[train]`
extra and `torch==2.14.0`. Recorded package lists accompany the results; these
commands describe the tested configuration, not a fully locked installer.

Download the pinned policy and benchmark above into
`/workspace/checkpoints/pi05-jar` and `/workspace/data/maniguard-bench`.
Download the Open-Jev 2B package to `/workspace/checkpoints/openjev-2b` and its
Qwen 2B base at revision `15852e8c16360a2fea060d615a32b45270f8a8fc` into a shared
`HF_HOME=/workspace/.hf_home` cache. Keep this environment variable set for critic
serving. The checkpoint path is `openjev-2b/package/checkpoint`.

After accepting the upstream licenses, use OmniGibson's asset downloader for
robot assets and BEHAVIOR 3.7.2rc1 assets with
`OMNIGIBSON_DATA_PATH=/workspace/ManiGuard/behavior-1k/datasets`.
Download `IDEAS-Lab-Northwestern/franka-panda-longfinger` with **`--repo-type dataset`**, revision `5f65ec0a715bc88ea865c2c69133b08b395c66bc`, and copy its
contents into `omnigibson-robot-assets/models/franka/franka_panda_longfinger`
under that dataset path (create the directory first).

Start `scripts/remote/serve-pi05.sh` and `scripts/remote/serve-openjev-2b.sh` under
the node's process supervisor. Both bind localhost. The critic health response
must identify `Qwen/Qwen3.5-2B`; `reset-policy.py` validates the policy's served
configuration and checkpoint path and checks a finite 16×8 action result.
Provide `OPENROUTER_API_KEY` through the environment, or a private mode-0600
`/run/safetyjev/openrouter.key` file. Then run:

```bash
/workspace/conda/behavior51/bin/python -u \
  /workspace/SafetyJev/scripts/remote/planner-integration-sweep.py
```

The script uses the documented `/workspace` paths, resets policy sampling before
each case, and runs shadow, guard-only, and guard-plus-DeepSeek modes for at most
64 actions each on `task_0000/base`, seed 0. It refuses to overwrite an existing
case directory. Output is `artifacts/planner-integration/`; logs go to
`/workspace/logs/integration-{shadow,guard,planner}.log`. Total GPU memory is
sampled once per second. Guard cases use 0.35 **only to exercise the rejection
path**, with three regenerations and at most 20 planner requests. This is not a
calibrated threshold or a predictor-quality experiment.

A completed run must have a valid oracle and successful selected-candidate
scoring. Inspect `decisions.jsonl` and `planner.jsonl` to establish whether repair
was exercised and a replacement actually executed; completion alone does not
establish either. Preserve failed runs as well as successful ones. Back up raw
captures and videos, remove temporary credentials, and stop the test services
when finished. Rental-instance shutdown is a separate user decision.


## Trained visual classifier: base Jar runtime evaluation

The step-20,000 checkpoint in `IDEAS-Lab-Northwestern/SafetyJev-Checkpoints`
is a **current-state visual predicate classifier**, not an action-conditioned
forecast model. Use `safetyjev.visual_runtime`, not the text-only Open-Jev adapter
or the guard threshold. Its inputs are current overview/wrist RGB images and
one of five trained questions. No robot state, action chunk, AP truth, outcome,
or cumulative monitor rejection enters the classifier request.

The five questions cover jar tilt, floor level, lid closed, on support, and
open while off support. Yes is not always unsafe: a closed lid or supported jar
is normally desirable. Ground truth comes from the matching current AP values
at the image's simulation step, with the exact polarity/conjunction in
`configs/jar-visual-queries.json`. A past temporal violation does not force later
frame labels to remain positive.

The pinned release is `7e4a72d3d2460612f5d3693a9676138bed678a5f`, subdirectory
`round2-amd-20k/models/step-20000`. Download that directory, `calibration.json`,
and `SHA256SUMS`; verify all ten model/calibration files. Its base is
`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Use the visual loader from `https://github.com/666harrypeng/Open-Jev.git`, commit
`533536be8cdabe9e0c47011f2f4d1397b87895ce`. The loader's SHA-256 is
`288416a9025441c169b92b8ee4ef0e8c67536630ae0b52f4fd77e05dc3aeb8c6`, matching
the release's training identity. The old text-only upstream loader discards the
vision tower and is not a substitute.

The new node keeps this loader at `/workspace/Open-Jev-visual-pinned`, with its
own `.venv`: Python 3.11, `torch==2.14.0`, `torchvision==0.29.1`,
`transformers==5.10.2`, `peft==0.19.1`, `numpy==1.26.4`, and `pillow==11.0.0`.
Install its `.[train]` extra. Set `HF_HOME=/workspace/.hf_home` during download
and serving. After all pinned resources are cached, serving can use
`HF_HUB_OFFLINE=1`. Credentials belong in temporary runtime storage and must not
be included in source or artifacts.

Start the fine-tuned policy as above. In the visual environment, with
`PYTHONPATH=/workspace/SafetyJev`:

```bash
python -m safetyjev.visual_server \
  --checkpoint /workspace/checkpoints/safetyjev/round2-amd-20k/models/step-20000 \
  --config /workspace/SafetyJev/configs/jar-visual-step20000.json \
  --calibration /workspace/checkpoints/safetyjev/round2-amd-20k/calibration.json
```

The endpoint binds only `127.0.0.1:8792`. It batches the five questions, performs
one warm-up on synthetic images, and exposes readiness only after model loading
and warm-up. Warm-up scores are excluded from evaluation. The client checks the
served checkpoint/base/loader identities and exact question definitions.
The checkpoint is loaded in BF16 with a frozen visual encoder and its trained
LoRA/head; it is not quantized or further trained during evaluation.

From the configured simulator environment:

```bash
python /workspace/SafetyJev/scripts/remote/jar-visual-quick-eval.py
```

This fixed development plan uses `task_0000/base`, `task_0001/base`, and
`task_0002/base`, seed 0, 256 actions maximum per scene, and a sample every eight
actions including step 0. It resets policy sampling between cases, retains
failed/incomplete outcomes, records GPU memory every second, and does not alter
the policy. The output directory is
`artifacts/jar-visual-step20000-20261006`. It refuses to overwrite an existing run.
Scenes are base-only; training-group independence has not been established.

The simulator process stores `classification-inputs.jsonl` (question text,
image paths/hashes and request hash), `classification-labels.jsonl` (evaluator-only
current truth), and `classification-predictions.jsonl` (logits, raw/calibrated Yes
scores and latency), plus normal monitor traces and outcome files. No future
window labels are generated for these classifications. Raw images/videos stay
in the ignored artifact backup.

```bash
python -m safetyjev.visual_runtime report \
  --episodes artifacts/jar-visual-step20000-20261006/episodes \
  --output artifacts/jar-visual-step20000-20261006/report.json
```

Raw Yes probability is `sigmoid(logit)`. The release's calibration is
`sigmoid(logit / 3.95 + prior_log_odds[question_id])`; parameters are frozen before
these rollouts and are not refitted to their labels. Both are classified at 0.5.
Report TP/FN/FP/TN, Yes/No counts, recall, specificity/false-positive rate,
balanced accuracy, AUROC, and calibration diagnostics per question. Undefined
metrics for a missing class remain null. Recorded client latency covers the five-question HTTP round trip and inference,
but excludes PNG writing and request serialization before the timer. Dividing it
by five is a throughput
normalization, not independently measured single-question latency. Simulator
time pauses during calls, so this does not demonstrate a real-time controller.

The forecast-report path also now keeps global `__all__` queries out of
per-constraint headline, episode, and family metrics. Its global results remain
in `global_task_forecast`; all-request timing is in `all_query_latency_s`.
Historical saved reports have not been silently rewritten.

Audit the saved raw captures without restarting GPU services:

```bash
PYTHONPATH=. python scripts/remote/audit-visual-evaluation.py \
  artifacts/jar-visual-step20000-20261006 \
  --calibration /workspace/checkpoints/safetyjev/round2-amd-20k/calibration.json \
  --output artifacts/jar-visual-step20000-20261006/audit-summary.json
```

This verifies recorded alignment, image/request hashes, and score formulas;
it does not independently validate contact-based simulator labels.
