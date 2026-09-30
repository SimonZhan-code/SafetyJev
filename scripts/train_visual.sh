#!/usr/bin/env bash
set -euo pipefail
SAFETYJEV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAFETYJEV_ROOT"
SAFETYJEV_PYTHON="${SAFETYJEV_PYTHON:-$SAFETYJEV_ROOT/.venv-visual/bin/python}"
export HF_HOME="${HF_HOME:-$SAFETYJEV_ROOT/checkpoints/hf_cache}"
exec "$SAFETYJEV_PYTHON" -m safetyjev.visual_train "$@"
