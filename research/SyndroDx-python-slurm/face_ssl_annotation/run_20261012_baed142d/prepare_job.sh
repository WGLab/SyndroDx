#!/bin/bash
#SBATCH --job-name=local-face-input
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/prepare-%j.log
#SBATCH --error=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/prepare-%j.err
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

cd /opt/craniofacial/SyndroDx-python-slurm
/opt/craniofacial/user_home/miniconda3/envs/qwen-finetune/bin/python /opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/annotation_pipeline.py prepare
