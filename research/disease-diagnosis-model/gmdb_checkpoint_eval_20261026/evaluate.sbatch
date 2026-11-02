#!/bin/bash
#SBATCH --job-name=gmdb-cranio-frozen26
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_checkpoint_eval_20261026/slurm-%j.private.log
#SBATCH --error=/opt/craniofacial/disease-diagnosis-model/gmdb_checkpoint_eval_20261026/slurm-%j.private.log
set -euo pipefail
umask 077
cd /opt/craniofacial/disease-diagnosis-model/gmdb_checkpoint_eval_20261026
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python evaluate_checkpoint.py
