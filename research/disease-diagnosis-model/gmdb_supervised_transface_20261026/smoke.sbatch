#!/bin/bash
#SBATCH --job-name=gmdb-follow-on-smoke
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/logs/smoke-%j.private.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
export PYTHONPATH=/opt/craniofacial/disease-diagnosis-model/transface_arcface_identity_cpt_20261026/dependencies
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/smoke_test.py
