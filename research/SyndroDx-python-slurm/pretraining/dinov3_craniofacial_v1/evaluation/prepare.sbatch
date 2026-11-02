#!/bin/bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true WANDB_MODE=disabled DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/prepare_evaluation.py --world-size 17
