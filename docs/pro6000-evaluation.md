# Single-GPU preliminary evaluation — 2026-09-29

**Partial completion:** fine-tuned π0.5 plus each of four decision-model variants
passed joint serving tests on one RTX PRO 6000. Isaac Sim failed during startup,
so simulation never joined a successful run and no safety accuracy was measured.

## Measured serving results

Hardware: RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported VRAM,
driver 595.91.07, Ubuntu 24.04.4, 123 GiB host RAM.

π0.5 was **ManiGuard's jar-task fine-tuned checkpoint**, not stock π0.5:
`IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora`,
revision `1d84eda070313a202a595e449fcf41a1a1e8a546`, step `7400`.
Every probe checked the server's configuration and checkpoint path before inference.
The server used the checkpoint's bundled normalization statistics.

| Decision model | Successful timed requests | Median per-constraint HTTP latency | Median four-request sequence | Total GPU memory snapshot, including π0.5 |
|---|---:|---:|---:|---:|
| Published Open-Jev 2B | 20/20 | 83.7 ms | 343.8 ms | 13.7 GiB |
| Qwen 4B, untrained Open-Jev decision head | 20/20 | 122.1 ms | 515.2 ms | 18.0 GiB |
| Published Open-Jev 9B | 20/20 | 175.5 ms | 748.7 ms | 27.1 GiB |
| Published Open-Jev 27B v1.1 | 20/20 | 554.6 ms | 2358.4 ms | 60.6 GiB |

Each row used exactly the same saved fixture: synthetic robot state, blank camera
images for π0.5, one generated eight-action proposal, and the actual static
constraints from ManiGuard jar task 0000. No action was executed. The fixture
contains no oracle labels or monitor outcomes. SHA-256:
`1e63a5218be9be2b8fb4498320c31bf5479305ad038079de8339697d88da1dc8`.

The four requests are three individual constraints plus an independent
combined-task forecast. Each model received one warmup request per constraint,
then five repetitions of those four requests. π0.5 remained resident during all
decision-model requests and produced finite `(16, 8)` outputs at each variant's
start. Its three-request median was approximately 59–64 ms in these probes.

These are **sequential inference requests with both models resident**, not
concurrent inference streams, simulation timing, or a production benchmark.
The small fixed-input sample measures serving behavior, not task diversity.
GPU memory values are end-of-probe snapshots. The 4B/9B/27B sweep additionally
sampled total memory once per second during loading and requests; sampled maxima
were 18,430 / 27,760 / 62,102 MiB. This can miss short peaks and includes allocator
caches. The simulator was absent from every successful measurement.

Open-Jev used BF16, batch size 1, a 4096-token limit, disabled prefix caching, and
the PyTorch fallback for the missing optional fast linear-attention kernels.
No quantization, CPU weight offload, or additional GPU was configured.
Its loader retains the language backbone and discards the vision tower.

There are released Open-Jev 2B, 9B, and 27B packages. The 4B entry is explicitly a
different baseline: `Qwen/Qwen3.5-4B` initialized with Open-Jev's Yes-minus-No head,
without an Open-Jev adapter. The published 27B uses Qwen3.8 and a different
training mixture from the original 2B/9B releases. This is not a controlled
parameter-count ablation. No 14B fallback was needed because 27B loaded.

## Simulator blocker

The user explicitly accepted the Isaac Sim EULA and BEHAVIOR research-data license.
Isaac Sim 4.5.0.0, OmniGibson from ManiGuard's pinned BEHAVIOR submodule, BDDL,
Spot/BuDDy and Blackwell-compatible PyTorch 2.7.0+cu128 were installed.

Three bounded startup attempts were made:

1. The initial attempt reported missing GLU, Xt and Python requests dependencies,
   then segfaulted. Those dependencies were installed.
2. With those dependencies present, startup still segfaulted in the renderer/
   viewport initialization path.
3. Explicit single-GPU rendering with 256×256 resolution failed in the same path.

The logs also report that this Iray version does not support compute capability
12.0. That warning alone does not establish the crash's cause. Driver causality
has not been isolated. We did not alter the host driver or substitute a newer
simulator, which would require separate compatibility and benchmark validation.

[ManiGuard's installation documentation](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
reports repeatable failures above driver series 580 in its tested stack.
[NVIDIA's current compatibility notes](https://docs.omniverse.nvidia.com/dev-overview/latest/common/technical-requirements.html)
also describe some newer-driver/older-Kit incompatibilities, but do not directly
diagnose this exact 595.91.07 configuration.

The next full evaluation needs a working host/runtime combination; a 580.x RTX
node matching ManiGuard's validated setup was requested. The large BEHAVIOR asset
bundle and long-finger robot assets have not been downloaded on this node, since
even the empty simulator fails first. Benchmark scene definitions and review
videos for jar tasks 0000–0002 are downloaded.

## What this suggests for SafetyJev

- **Capacity:** 27B plus the fine-tuned VLA fits this GPU. That is promising
  capacity evidence, not proof of a three-component deployment.
- **Latency:** the current serial four-request 27B path takes about 2.36 s.
  An eight-step chunk at 20 Hz spans 0.4 s. Even the 2B sequence plus the roughly
  60 ms policy call consumes about that interval before simulator/perception
  overhead. Batch evaluation or a smaller model is worth testing before making
  runtime claims; synchronous simulated time does not impose a real robot deadline.
- **Probability interpretation:** in this one synthetic fixture, independently
  predicting any violation produced a score below an individual-clause score for
  both 2B and 27B. These outputs should not be treated as calibrated, mutually
  coherent probabilities. This is a diagnostic, not an accuracy result.
- **Observability:** the state-only input omits object poses, grasp/contact state,
  and images. Scaling parameter count does not supply the missing observations.
  A trained multimodal SafetyJev or an explicitly defined observable scene-state
  interface still needs to be designed and evaluated.

No model was fine-tuned on ManiGuard. The published Open-Jev models are general
decision models being tested without robotics safety fine-tuning.

## Reproduction and evidence

Pinned model/package revisions are in `configs/pro6000-models.json`.
Other source revisions:

- SafetyJev deployment: `89286cfd15907564dc240e9a73ac52407b3ddb10`, plus the
  serving probes introduced with this report.
- ManiGuard: `be97624e0acbec6b6f9260a08891b04168eb8e6c`.
- BEHAVIOR: `88454bd04f75dc57c00ab1f1a00bcde1ff505950`.
- openpi: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Open-Jev: `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`.

Raw JSON reports, the identical-input fixture, environment versions, sampled
memory traces, and crash-log hashes/excerpts are under
`docs/results/2026-09-29-pro6000/`. No safety labels or accuracy metrics are fabricated.

The real API test discovered that `model_id=Open-Jev-2B` is rejected by the
server. The example configuration and runbook now use its advertised identity,
`Qwen/Qwen3.5-2B`; the separate predictor revision identifies the Open-Jev
adapter/head package. The rejected-request probe is retained as failure evidence
and excluded from the table above.

On the configured node, start the π0.5 service and one decision service, then run
`scripts/remote/joint-model-probe.py` with that server's model identity.
The saved `run-model-sweep.py` stops the 2B service and tests 4B/9B/27B sequentially.
It expects the recorded `/workspace` layout and preconfigured Supervisor services;
it is a reproduction script for this node, not a general provisioner.

All model services were stopped after testing. The rental instance was not
stopped or destroyed. Files under its container-backed `/workspace` do not
survive instance destruction; result copies were retrieved locally.
