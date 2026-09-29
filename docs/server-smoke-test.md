# Rented GPU smoke test — 2026-09-29 UTC

Status: policy serving verified; simulator rollout and safety evaluation **not run**.

## Server

- SSH: `ssh -p <SSH_PORT> root@<SERVER_IP>` (use the instance connection details)
- Ubuntu 24.04.4; GPU reports RTX 4090, 49,140 MiB total memory.
- Host NVIDIA driver: 615.71.09. This is outside ManiGuard's validated 580-series
  setup. Successful Vulkan/CUDA checks do not establish Isaac Sim compatibility.
- `/workspace` is on the container overlay, not a separate persistent volume.
  Keep results off the instance before recycling or destroying it.

## Verified

- SafetyJev copied to `/workspace/SafetyJev`; all 24 CPU tests pass.
- Pinned ManiGuard source hooks verified.
- Missing rendering libraries installed with the provider's
  `install-display-drivers` helper. No host driver was changed.
- `vulkaninfo --summary` detects the GPU when
  `LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64` is set.
- PyTorch 2.6.0+cu124 GPU matrix operation passes; JAX detects a CUDA device.
- Spot/BuDDy compile a two-state monitor for `G !violation`.
- π0.5 jar checkpoint step 7400 loads and answers three synthetic observations
  with finite `(16, 8)` action arrays. These inputs contain blank camera frames;
  this is a serving test, not a robotic or safety evaluation.

| Request | End-to-end latency |
|---|---:|
| First, including compilation | 16.410 s |
| Second | 0.111 s |
| Third | 0.107 s |

Two warmed requests are insufficient for a performance benchmark. The standalone
policy process occupied approximately 8.4 GiB in an `nvidia-smi` snapshot. There
was no simulator or safety predictor running concurrently.

Raw evidence: `docs/results/2026-09-29/synthetic-policy-smoke.json` and
`docs/results/2026-09-29/runtime-prerequisites.json`.

## Reproducible resources and paths

- ManiGuard: `/workspace/ManiGuard`, commit
  `be97624e0acbec6b6f9260a08891b04168eb8e6c`.
- BEHAVIOR submodule: `88454bd04f75dc57c00ab1f1a00bcde1ff505950`.
- openpi: `/workspace/openpi`, commit
  `215abfb217dbac7d5f1273282331b9b1866c0479`; installed with its frozen uv lock,
  Python 3.11, without the dev group. This revision passed the policy smoke test;
  the original checkpoint's training-code revision has not been established.
- π0.5 weights: `/workspace/checkpoints/pi05-jar/7400`, HF revision
  `1d84eda070313a202a595e449fcf41a1a1e8a546`.
- One benchmark scene downloaded under
  `/workspace/data/maniguard-bench/jar_transport/task_0000/base`, HF revision
  `2ea32a1451669fb736ae78ffce9cc82aad4cceac`.
- Simulator prerequisite environment: `/workspace/conda/behavior`, Python 3.10,
  Spot, BuDDy, PyTorch cu124, NumPy <2 and PyAV. Isaac Sim and the BEHAVIOR
  asset bundle have not been installed.
- Separate minimal test environment: `/workspace/SafetyJev/.venv`.
- An earlier CUDA-only test environment also exists at `/workspace/venvs/behavior`;
  use the conda environment above for the simulator going forward.

## Service and next step

The managed policy service `safetyjev-pi05` binds only to `127.0.0.1:8000` and was
**stopped after testing**. It does not autostart. The rented instance itself was
not stopped or destroyed.

```bash
supervisorctl start safetyjev-pi05
/workspace/openpi/.venv/bin/python /workspace/SafetyJev/remote/policy-smoke.py
supervisorctl stop safetyjev-pi05
```

Logs: `/workspace/SafetyJev/artifacts/pi05-server.log`. The corresponding scripts
are saved locally under `scripts/remote/` and remotely under
`/workspace/SafetyJev/remote/`.

The upstream `behavior-1k/setup.sh` explicitly prompts for acceptance of NVIDIA's
Isaac Sim EULA and BEHAVIOR's non-commercial academic research data license.
User confirmation was requested and remains pending. Neither acceptance flag
has been supplied. Once confirmed, install the simulator, test headless startup
on driver 615.71, and only then download the large licensed scene assets and run
the real jar rollout. If startup fails because of the host driver, use a host
with ManiGuard's tested driver configuration; do not change the container's
host-injected NVIDIA driver.

No Open-Jev inference, SafetyJev fine-tuning, real rollout, or safety accuracy
measurement was performed during this smoke test.
