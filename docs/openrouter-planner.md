# System 1: OpenRouter instruction repair

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

## What the planner receives

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

## Output and execution semantics

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

## Configure and run

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
[guarded-loop guide](guarded-loop.md). The node-specific
`scripts/remote/planner-580-2b-isaac51.sh` launches a 64-action smoke in the existing
experimental Isaac 5.1 environment and requires both environment variables above.
It assumes the policy and guard services are already running.

## Artifacts and comparisons

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

## Validation — 2026-09-29

All 53 CPU tests passed, including the actual pinned ManiGuard execution loop and
its openpi observation-to-prompt mapping with controlled model/simulator boundaries.
Tests verify revised-prompt delivery, rechecking every replacement, no planner
calls on initial acceptance, bounded calls, timeout/error handling, image request
construction, schema validation, oracle isolation, and credential-free artifacts.

No live OpenRouter request or GPU simulator evaluation was performed for this
implementation. An explicit model, runtime API key, and available simulator node
are needed for that validation. The previously used GPU node refused SSH during
the preceding guard-loop implementation.

API implementation follows the official [OpenRouter API reference](https://openrouter.ai/docs/api/reference/overview)
for authentication, chat completion requests, and image content, and
[structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs)
for JSON schema responses and `provider.require_parameters`.
