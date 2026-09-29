#!/bin/bash
set -euo pipefail
export PYTHONPATH=/workspace/ManiGuard
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ManiGuard
exec /workspace/openpi/.venv/bin/python -u -m maniguard.serve.openpi_native --config pi05-base_datagen_v1_jar_joint_2cam_lora --checkpoint /workspace/checkpoints/pi05-jar/7400 --host 127.0.0.1 --port 8000
