#!/bin/bash
#SBATCH --job-name=cranio-qwen32b-diagnosis
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/cranioface_qwen_alignment_20261027/logs/followon-%j.private.log
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/cranioface_qwen_alignment_20261027/followon.py
