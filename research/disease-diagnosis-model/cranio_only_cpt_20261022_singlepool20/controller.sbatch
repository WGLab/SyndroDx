#!/bin/bash
#SBATCH --job-name=cranio-singlepool-coord
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20/controller_%j.private.log
#SBATCH --error=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20/controller_%j.private.log
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_singlepool20
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python pool_controller.py
