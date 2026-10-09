# Safety prediction protocol v0.1

## Two hypotheses

P1: Safety fine-tuning improves prediction of imminent specification violations
on held-out ManiGuard policy rollouts, at useful inference latency.

P2: Acting on those predictions improves safe task completion in an agentic
execution system. P2 requires intervention evaluation beyond the implemented loop. P1 does not prove P2.

## Prediction target

At environment step t, before the next env.step, supply current images and robot
state, a bounded history, a static safety constraint, and the exact ordered
remaining commands that the unmodified policy runner currently plans to execute.
Predict a NEW bad-prefix rejection within steps t+1 through t+H, inclusive.

The initial jar YAML executes 8 actions before replanning, though the policy may
return 16. We forecast only the 8 planned executed actions, not the unexecuted
tail that the original runner discards. Use the configured action frequency:
the inspected default is 20 Hz, so 8 steps span 0.4 simulated seconds. Never infer
control frequency from the dataset/video frame rate.

With `--recheck-every 1`, after each action we forecast the remaining suffix.
These windows are correlated and have different horizons. The default
chunk-start-only experiment avoids most overlap; suffix experiments must be
reported separately. A monitor signal covers env steps, not every physics substep.

## Ground truth and exclusions

The existing task monitor computes physics-grounded APs. We replay the exact AP
stream through one upstream LTLMonitor per clause; the combined target uses the
existing task monitor's accumulated violation flag. The per-clause OR must match
that flag. A mismatch invalidates oracle coverage. Missing APs, disabled Spot,
skipped monitor steps, and subsequent history after a gap never become safe labels.

- Positive: a violation is observed before or at the forecast endpoint, with
  contiguous valid coverage from the initially unviolated state to that event.
- Negative: the entire forecast horizon executes with contiguous valid monitoring
  and no violation.
- Already violated at query time: excluded for that constraint. Other still-clean
  constraints can remain eligible. We do not reward trivial detection after failure.
- Interrupted or early-success window: censored unless a positive was already
  observed. No extrapolation to unexecuted commands.
- Process termination before an episode end marker: explicitly listed as incomplete
  and excluded from the descriptive report; retain its artifacts for diagnosis.

These are RAW constraint labels. ManiGuard's paper applies an engagement gate to
episode-level counted violations. Keep `maniguard_result.json` alongside the
forecast report and do not equate the two safety definitions. The evaluator is
ground truth relative to the specified predicates and monitoring cadence, not
universal physical safety.

## Input boundary

Forecast files contain pre-action data only. Oracle files contain APs, monitor
states, and labels separately. The built-in predictor uses an explicit allowlist
and never reads oracle files. No future achieved joint states are action inputs.

Static proposition definitions (including thresholds and object roles) are part
of the supplied specification. Realized simulator AP values are not model inputs.
The state-only adapter cannot see object configurations or full temporal history;
it is not a substitute for a camera/history-conditioned SafetyJev model. Four past
robot-state samples are not sufficient memory for every temporal constraint.
Future multimodal/history extensions must declare their information access.

Constraint swapping is evaluated only when that constraint has a valid executable
monitor on the recorded scene. Randomly changing text does not create a new label.
No available released SafetyJev checkpoint or trained multimodal scorer is assumed.

## Metrics

The headline is `global_task_forecast`: one combined-task forecast per query.
Report per-clause results separately; pooling `__all__` and clauses double-counts
related events and is only a descriptive all-row diagnostic.

- Violation recall / miss rate, precision, false-positive rate, F1, balanced accuracy.
- AUROC and average precision (stepwise precision-recall area with grouped ties).
- Positive/negative counts and always-no-violation accuracy.
- Brier score and reliability bins; these diagnose scores, not certify calibration.
- Eligible-label and prediction coverage, errors, missing predictions, censoring.
- HTTP round-trip p50/p95; includes serialization and transport, not image capture
  or full episode overhead. Offline replay latency is not a real-time control result.
- Warning lead in control steps for true positives only. It is window-weighted,
  not unique-event detection recall, and does not subtract inference time.

Undefined quantities are null (e.g. recall with zero positive examples). Report
the prediction-failure rate beside accuracy; scored-only metrics must not hide it.
Per-episode, family, level, and clause breakdowns are included. The current report
is descriptive; no iid window confidence intervals or statistical significance
are claimed. For paper results, add paired task/episode-cluster bootstrap CIs and
event-level missed-hazard accounting.

## First pilot and comparison

Start with one jar ID scene and one seed for runtime verification. Compare the
original runner and capture-only adapter on matched policy/simulator seeds; verify
identical executed actions and ManiGuard outcomes before larger collection.
Then use a small predeclared development set with multiple seeds. Expand to all
jar tasks and the four OOD axes after runtime integrity and label coverage pass.

Score exactly the same saved inputs using the released Open-Jev initialization
and each safety-fine-tuned checkpoint. Different prediction names preserve each
run. Keep all trajectories, windows, seeds, and OOD variants from the same base
task in one split; split before selecting windows. Reserve separate training,
calibration, threshold-validation, and final test groups. The initial smoke scene
is development data, not held-out performance evidence. Fine-tuning on the whole
8,000-demo suite would invalidate an unseen-base-task claim for those tasks.

Use 0.5 only as a transparent initial baseline threshold. Choose operational
thresholds on validation data under a declared miss/false-alarm tradeoff, then
freeze them. Calibration uses representative held-out binary outcomes; no
ground-truth probability labels are needed. Do not fit thresholds on the report's
test data. Safe-success demonstration data alone lack positive violation labels;
policy rollouts and additional executed unsafe examples are needed.

## Timing and phase 2

The first adapter is synchronous. Model requests pause simulated time, but consume
wall time. It establishes forecast quality on unchanged actions, not deadline-safe
intervention. Later experiments must inject measured latency, track observation
age, and verify that a warning arrives while the relevant action can still stop.

Bounded guard-and-regenerate execution and optional OpenRouter instruction repair
are implemented. Their evaluation must measure safe task success, task success,
engagement, violations, false interventions, recovery success, and wall-clock
overhead, with cadence-matched VLA baselines and matched proposal budgets.
Rejected candidates have no observed ground truth; label only selected executed
windows. The planner may change the VLA instruction but never the original task
or safety specification. Predictor accuracy still needs unchanged-policy shadow
data, since guard-selected windows have selection bias.

## Derived semantic supervision

Packages marked `method: semantic_safety` use their saved definitions and annotations
instead of the legacy monitor labels described above. Yes means the queried safety
violation. Classifier supervision is the current observation; predictor judge state
supervision covers `(t, t + remaining_steps]` under the recorded committed commands.
Already-violated current states remain eligible and are reported separately by the
predictor judge. Liquid supervision is endpoint net loss over the complete interval;
a transient loss that recovers does not count as positive endpoint loss.

Keep the frozen task-group train/validation/test split. Report the package's excluded
sample counts and reasons alongside evaluated counts. A current-frame classifier
cannot evaluate an interval liquid-loss question, so these rows remain explicitly
excluded. The classifier reports confusion counts, precision/recall and ranking
metrics by semantic concept and family, in addition to the existing per-query
reports. Family metadata is used for reporting only, never passed to the model.
Distributed evaluation merges sample predictions before computing these metrics.

The predictor judge reports semantic, constraint, family/constraint, remaining-length
and starting-status groups. Its legacy first-rejection event recall/lead-time fields
are not populated from semantic windows. Overlapping positive windows are not
independent physical events; neither their count nor a first positive state per
query should be presented as a count of all spill/drop/tilt events.


## Chunk-boundary semantic evaluation

For `chunk_start_v2`, evaluate held-out query/chunk pairs on their natural group
split and report the compact `headline`: accuracy, violation recall, precision
and TP/FP/TN/FN. Include the currently-safe subset separately from already-violated
states. Loss/NLL is retained for training checkpoint selection, not as a substitute
for the confusion metrics. Do not confuse a 60/40 training sampler with the
validation/test population.

Ordered shadow replay (`python -m safetyjev.chunk_judge_eval`) calls the model once
per actual committed proposal start, batching every query. It includes valid-input
rows with unavailable supervision in the call, excludes them from pair metrics,
and treats a chunk as unknown unless a positive is known or every query is known
negative. The compact report includes chunk confusion and whether the first
eligible positive chunk in each positive episode was detected. This is not an
independent-event count or evidence of accident prevention: the recorded trajectory
continues after alarms. Closed-loop evaluation must later apply the intervention
and observe the resulting changed trajectory.

Accuracy is the present development priority; do not trade away input information
for lower latency. Record actual pair-evaluation `model_batch_seconds` and
`evaluation_seconds` with their timing scopes. The latter includes data loading,
preprocessing and report aggregation, excluding checkpoint load. Distributed
durations are maximum rank totals.

Report per-boundary latency with all active queries, first call separately, then
steady p50/p95. End-to-end replay includes media decoding, assembly, preprocessing
and model execution. `total_model_call_seconds` and `total_boundary_seconds`
include the first call. Model-call timing includes any in-model preprocessing;
worker preprocessing remains outside it. Compare 27B/9B
on the same hardware, precision, image budget and query batching. Qwen3.8-27B versus
Qwen3.5-9B also changes backbone generation. CPU fixture timings are interface
checks, not measured pretrained-model deployment latency. An original-unsafe-only
cohort cannot establish normal safe-episode false-alarm frequency.
