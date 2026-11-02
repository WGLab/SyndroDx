#!/bin/bash
#SBATCH --job-name=cf-dinov2
#SBATCH --partition=gpu-xe9680q
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/extract_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
mkdir -p logs
PY=$HOME/miniconda3/envs/qwen-finetune/bin/python
nvidia-smi
$PY extract_features.py "$@"
