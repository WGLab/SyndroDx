#!/bin/bash
#SBATCH --job-name=cranio-single-h100
#SBATCH --partition=gpu-xe9680q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --no-requeue
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20/lease_%j.private.log
#SBATCH --error=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20/lease_%j.private.log
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python lease.py "$1"
