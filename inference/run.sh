#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUNTIME_PYTHON="$PYTHON_BIN"
elif [[ -x "$BUNDLE_DIR/.venv/bin/python" ]]; then
  RUNTIME_PYTHON="$BUNDLE_DIR/.venv/bin/python"
else
  RUNTIME_PYTHON=python3
fi
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
exec "$RUNTIME_PYTHON" "$BUNDLE_DIR/predict.py" "$@"
