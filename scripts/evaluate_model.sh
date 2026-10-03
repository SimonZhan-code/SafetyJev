#!/usr/bin/env bash
set -euo pipefail
SAFETYJEV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAFETYJEV_ROOT"
SAFETYJEV_PYTHON="${SAFETYJEV_PYTHON:-$SAFETYJEV_ROOT/.venv-visual/bin/python}"
export HF_HOME="${HF_HOME:-$SAFETYJEV_ROOT/checkpoints/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  exec "$SAFETYJEV_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node "$NPROC_PER_NODE" -m safetyjev.evaluate_trained "$@"
fi
exec "$SAFETYJEV_PYTHON" -m safetyjev.evaluate_trained "$@"
