#!/bin/bash
#SBATCH --job-name=cranio-geometry-audit
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_resume20/geometry_%j.private.log
#SBATCH --error=/opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_resume20/geometry_%j.private.log
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model/cranio_only_cpt_20261022_resume20
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python geometry_audit.py
