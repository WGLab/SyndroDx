#!/bin/bash
#SBATCH --job-name=arcface-packed-cache
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/transface_arcface_identity_cpt_20261026/logs/packed-cache-%j.private.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/transface_arcface_identity_cpt_20261026/performance_revision_02/build_packed_cache.py
