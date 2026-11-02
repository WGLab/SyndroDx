#!/bin/bash
#SBATCH --job-name=cranio-vitl-cache
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/transface_vitl_cpt_20261026/logs/cache-%j.private.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true DO_NOT_TRACK=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/transface_vitl_cpt_20261026/cache_inputs.py
