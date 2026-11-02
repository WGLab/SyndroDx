#!/bin/bash
#SBATCH --job-name=cranio-qwen-source-audit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/cranioface_qwen_alignment_20261027/logs/source-audit-%j.private.log
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/cranioface_qwen_alignment_20261027/audit_sources.py
