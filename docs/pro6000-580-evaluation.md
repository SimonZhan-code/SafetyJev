# RTX PRO 6000, driver 580 — compatibility evaluation

This run separates hardware capacity, simulator compatibility, and safety-model
quality. The original Isaac Sim 4.5 stack starts on this node, but produces
severely noisy camera observations on Blackwell. A separate Isaac Sim 5.1
compatibility environment produces clean images. Its benchmark equivalence is
not established; these are development diagnostics, not official benchmark scores.

## Node and fixed resources

- RTX PRO 6000 Blackwell **Server Edition**, 97,887 MiB VRAM.
- Host driver 580.126.09; `nvidia-smi` advertises CUDA 13.0.
- Ubuntu 24.04.4, approximately 204 GiB RAM, 256 GB container storage.
- Simulator PyTorch 2.7.0+cu128; separate policy and critic environments.
- Same ManiGuard fine-tuned jar π0.5 checkpoint, HF revision
  `1d84eda070313a202a595e449fcf41a1a1e8a546`, step 7400. No stock π0.5 substituted.
- Same ManiGuard code `be97624e0acbec6b6f9260a08891b04168eb8e6c`, same benchmark
  revision `2ea32a1451669fb736ae78ffce9cc82aad4cceac`, jar task 0000/base, seed 0.
- Required long-finger Franka artifact revision
  `5f65ec0a715bc88ea865c2c69133b08b395c66bc`.
- Open-Jev models remain the public, non-robotics-fine-tuned baselines. The 4B
  model has an untrained Yes-minus-No head. Model pins: `configs/pro6000-models.json`.

## Original stack: starts, but fails visual validation

Isaac Sim 4.5.0.0 + pinned OmniGibson 3.7.2 completed the empty headless test and
then executed a real jar rollout alongside π0.5 and Open-Jev 2B. Visual inspection
of both overview and wrist PNGs found heavy speckled noise. The rollout was
stopped at step 808; it is incomplete and excluded from performance reporting.
There were 809 valid oracle samples, 407 successful critic responses, and a first
raw safety violation at step 661. The final forecast window may be partial.

This is a separate issue from the previous node's startup crash. NVIDIA documents
that Blackwell on Linux lacks DLSS Ray Reconstruction with Kit versions below
106.5.3, which produces noisy real-time images. The BEHAVIOR maintainer explicitly
states that Blackwell is unsupported on its Isaac 4.5 track and recommends 5.0+:

- [NVIDIA renderer requirements](https://docs.omniverse.nvidia.com/dev-overview/latest/common/technical-requirements.html)
- [BEHAVIOR maintainer response](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2270#issuecomment-4850817820)

Driver 580 is therefore insufficient on its own to validate Blackwell for the
pinned ManiGuard simulator. Check actual policy-camera images before a long run.

## Separate compatibility stack

The original installation and checkout were preserved. The experimental environment
uses Python 3.11, Isaac Sim 5.1.0.0, and upstream OmniGibson 3.8.0 from commit
`d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864` (`feat/isaac-5.0`, which supports 5.1).
That branch retains ManiGuard's expected legacy robot API and uses the same
3.7.2rc1 scene assets. No host-driver change or modification to ManiGuard's
runner/monitor source was made.

An initial attempt with upstream commit `dc0ee5be6a70ae0bc4a39cb1f01d28d7791bf88d`
failed before scene creation because its refactored robot API removed
`omnigibson.robots.manipulation_robot`. The selected branch also required its
declared evaluation extras, including the pinned LeRobot package.

The clean-camera run's initial monitor state was safe, but its first raw
close-before-lift violation occurred at step 3. This differs from the original
stack and requires physics, reset, and observation parity checks before treating
results as comparable. Both stacks also logged a convex-collision-mesh warning
for the jar's label geometry. Keep those warnings in the diagnostic record.

The policy is warmed with a different episode seed before repeated real rollouts.
ManiGuard's server reseeds only when the requested seed changes; without this,
repeating the same scene/seed on one server continues the prior random sequence.
The benchmark itself still receives seed 0 and derives episode seed 1528289969.

## Interpretation

Safety scores are forecasts of a binary bad-prefix event over the next executed
action chunk. No accurate ground-truth probability is assumed. Already violated
constraints and incomplete windows are excluded by the existing labeling rules.
An early combined-task violation can leave very few eligible global forecasts.
Report counts and event times alongside any metrics; correlated windows from one
episode cannot establish general accuracy or calibration.

The critics consume robot state, recent robot states, action chunks, instruction,
and static constraints. They do not receive camera pixels, simulator object-state
truth, oracle samples, or outcome labels. This is an information-limited baseline.

## Completed 2B pilot

The compatibility run completed all 2,000 actions, with 2,001 valid monitor samples
and 1,000 successful online forecasts (three constraints plus the global query
at each of 250 chunk starts). The task did not succeed; raw LTL safety was violated.
At threshold 0.5, 2B missed both observed per-constraint events. There were 282
eligible per-constraint windows: 280 negatives and two positives. The separate
global forecast had only one eligible window, which was positive and missed.
These counts are too small and imbalanced to establish model quality.

The 283 eligible rows including the separate global query are replayed identically
for the other model sizes. The new `predict --eligible-only` flag applies the
existing exclusion rules in the evaluator, then passes only original pre-action
forecast records to the scorer. No oracle fields are inserted in requests. The
other 717 already-violated rows remain captured but are omitted from replay.
Selection is recorded in prediction metadata. Online timing includes all 1,000
requests; offline replay timing uses the eligible subset, so their medians are
not a controlled serving-speed comparison.

## Joint execution results

| Critic | Executed actions | Successful online requests | Median request latency | Sampled peak total GPU memory |
|---|---:|---:|---:|---:|
| 2b | 2,000 | 1,000 | 195.2 ms | 18.1 GiB |
| 4b-base | 64 | 32 | 249.5 ms | 22.4 GiB |
| 9b | 64 | 32 | 269.4 ms | 31.5 GiB |
| 27b | 64 | 32 | 733.3 ms | 65.1 GiB |

Memory includes the policy, critic, and simulator. The 2B run is a full pilot;
the other rows are 64-action compatibility checks. Each other model also completed
283/283 eligible-window replay requests with no errors. No 14B fallback was needed.

At threshold 0.5, every critic missed both raw per-constraint positive windows
and predicted all 280 negative windows below threshold. The resulting 99.3%
per-constraint accuracy merely matches the always-no-violation baseline. It is
not evidence of useful violation detection.

The first raw violation (step 3) preceded the first robot contact (step 160).
ManiGuard records `ltl_violated=true` but **`counted_violation=false`** under its
engagement-gated outcome, and `ever_grasped=false`. Our raw-window target is
therefore different from its engagement-gated episode safety statistic. Both
values are retained in the result JSON; neither is substituted for the other.

At the configured 20 Hz, eight actions represent 400 ms of simulated time.
The 2B median request alone is about 195 ms; four serial constraint queries
already exceed that interval. This pilot establishes compatibility, not a
real-time safety-feedback deadline.

All evaluation services were stopped after auditing; GPU memory returned to
0 MiB. The rental instance itself was not stopped or destroyed. A full local
backup of captures, images, videos, and logs is in the ignored
`artifacts/pro6000-580/` directory. Summary JSON, memory samples, and capture
hashes are committed under `docs/results/2026-09-29-pro6000-580/`.

Validation: 26 CPU tests pass, source hooks verify against pinned ManiGuard,
all four joint-run captures have valid monitor records and successful requests,
and replay IDs exactly equal the same 283 eligible forecast IDs.

## Reproduction and limits

The node-specific scripts expect `/workspace/ManiGuard`, `/workspace/openpi`,
`/workspace/Open-Jev`, the recorded checkpoints, and Supervisor services named
`safetyjev-pi05` and `safetyjev-openjev-{2b,4b-base,9b,27b}`. They bind inference
only to localhost and do not autostart. The new simulator environment is
`/workspace/conda/behavior51`; its editable OmniGibson checkout is
`/workspace/BEHAVIOR-5.1`. Set `OMNIGIBSON_DATA_PATH` to the original downloaded
assets; do not silently substitute newer assets.

Use `configs/jar-isaac51-provenance.json` with
`scripts/remote/capture-580-2b-isaac51.sh`, after starting π0.5 and 2B and running
`reset-policy.py`. The sweep requires that completed pilot, then runs each other
model for 64 actions and replays the 283 eligible inputs. `audit-580-results.py`
checks complete captures, valid monitors, successful predictions, and identical
replay IDs, then records file hashes and one-second GPU memory samples.
The audit's first-pilot timestamp is specific to this recorded run.

Each safety request blocks action execution during this shadow experiment.
Models and simulator share one GPU, but these are interleaved calls, not a
measurement of simultaneously executing inference kernels. Simulated action
frequency is not a guarantee of real-time wall-clock control. GPU memory sampling
can miss transient peaks. No quantization or CPU weight offload was configured.

The 4B head is untrained; 27B uses a different Qwen generation and training mixture.
This is not a controlled size ablation. No SafetyJev fine-tuning was performed.

Repeated simulator processes with the same scene and policy seed did not produce
bitwise-identical first-64-step trajectories: maximum recorded action differences
versus the 2B run were approximately 0.0324 (4B), 0.0247 (9B), and 0.0174 (27B).
These short runs establish coexistence and API operation, not a causal comparison
of safety outcomes across critics. Accuracy comparisons use the identical saved
forecasts instead. Instrumented-versus-original trajectory parity remains unverified.
