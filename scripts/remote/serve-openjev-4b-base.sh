#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
cd /workspace/Open-Jev
exec /workspace/Open-Jev/.venv/bin/python -u -m jev.server --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8791
