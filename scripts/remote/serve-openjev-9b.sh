#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
cd /workspace/Open-Jev
exec /workspace/Open-Jev/.venv/bin/python -u -m jev.server --checkpoint /workspace/checkpoints/open-jev-9b/package/checkpoint --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8791
