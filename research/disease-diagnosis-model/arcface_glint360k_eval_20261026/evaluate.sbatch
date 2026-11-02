#!/bin/bash
#SBATCH --job-name=gmdb-unadapted-arc26
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/arcface_glint360k_eval_20261026/slurm-%j.private.log
#SBATCH --error=/opt/craniofacial/disease-diagnosis-model/arcface_glint360k_eval_20261026/slurm-%j.private.log
set -euo pipefail
umask 077
cd /opt/craniofacial/disease-diagnosis-model/arcface_glint360k_eval_20261026
export PYTHONPATH=/opt/craniofacial/disease-diagnosis-model/arcface_glint360k_eval_20261026/dependencies
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python benchmark.py
