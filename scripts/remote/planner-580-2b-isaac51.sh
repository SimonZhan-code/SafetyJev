#!/bin/bash
set -euo pipefail
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in the runtime environment}"
: "${OPENROUTER_MODEL:?Set an explicit OpenRouter model ID}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/workspace/ManiGuard/behavior-1k/datasets
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONNOUSERSITE=1
export PYTHONPATH=/workspace/SafetyJev:/workspace/ManiGuard
cd /workspace/ManiGuard
exec /workspace/conda/behavior51/bin/python -u -m safetyjev.cli capture \
 --maniguard-root /workspace/ManiGuard \
 --output /workspace/SafetyJev/artifacts/jar-planner-580-2b-isaac51 \
 --provenance /workspace/SafetyJev/configs/jar-isaac51-provenance.json \
 --online-predictor /workspace/SafetyJev/configs/openjev-state-only.json \
 --planner-config /workspace/SafetyJev/configs/openrouter-planner.json \
 --execution-mode guard_regenerate --guard-threshold 0.5 --max-regenerations 3 \
 -- --config configs/eval/jar_transport_joint.yaml \
 --benchmark-root /workspace/data/maniguard-bench/jar_transport \
 --scenes task_0000/base --seed 0 --max-steps 64 --tag safetyjev-planner-580-2b-isaac51
