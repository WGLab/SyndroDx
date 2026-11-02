#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=cranio-paired-eight
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=09:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/controlled_cpt_20261021_parallel/controller.private.log
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true WANDB_MODE=disabled DO_NOT_TRACK=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/controlled_cpt_20261021_parallel/controller.py
