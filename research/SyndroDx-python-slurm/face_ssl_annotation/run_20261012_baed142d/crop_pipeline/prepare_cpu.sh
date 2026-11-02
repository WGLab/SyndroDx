#!/bin/bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true DO_NOT_TRACK=1 OMP_NUM_THREADS=1
CROP_STAGE_DIR=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/crop_pipeline
cd "$CROP_STAGE_DIR"
exec "$CROP_STAGE_DIR/venv/bin/python" -u "$CROP_STAGE_DIR/prepare_crops_cpu.py"
