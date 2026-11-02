#!/bin/bash
set -euo pipefail
umask 077
CROP_ROOT=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/crop_pipeline
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
cd "$CROP_ROOT"
for attempt in $(seq 1 360); do
    if test -f READY.json; then
        exec "$CROP_ROOT/venv/bin/python" "$CROP_ROOT/crop_worker.py" --slot "$1"
    fi
    sleep 20
done
printf '%s\n' 'Crop environment readiness timeout; allocation released.' >&2
exit 1
