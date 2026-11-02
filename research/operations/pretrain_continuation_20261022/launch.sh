#!/bin/bash
# Run only after scratch write access is granted. This has NOT been submitted.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export SLURM_CONF=/opt/site/slurm/slurm.conf
export PYTHONPATH=/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1
BASE=/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1
PHASE="$BASE/continuation_20261022_convergence"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$BASE/venv/bin/python" -B "$HERE/continue_pretrain.py"
"$BASE/venv/bin/python" -B "$HERE/continue_pretrain.py" --deploy
# The coordinator belongs to a persistent terminal, independent of this chat.
# GPU allocation cleanup is handled by both coordinator and lease heartbeats.
/usr/bin/tmux -S "$HERE/coordinator.sock" new-session -d -s pretrain-continuation \
    "$BASE/venv/bin/python -B '$PHASE/continue_pretrain.py' --coordinate > '$PHASE/coordinator.private.log' 2>&1"
"$BASE/venv/bin/python" -B "$HERE/continue_pretrain.py" --status
