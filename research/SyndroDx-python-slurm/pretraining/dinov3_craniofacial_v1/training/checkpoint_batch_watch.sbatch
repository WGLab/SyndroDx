#!/bin/bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true DO_NOT_TRACK=1
exec /usr/bin/python3 /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/checkpoint_batch_handoff.py
