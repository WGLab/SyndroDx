#!/bin/bash
#SBATCH --job-name=cf-base
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/base_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
PY=$HOME/miniconda3/envs/qwen-finetune/bin/python
# same architecture the SSL run adapts, so the eventual comparison is apples-to-apples
$PY -u extract_features.py --model facebook/dinov2-base \
    --out features_base.npy --batch 256 --workers 2
$PY -u probe_audit.py --features features_base.npy --skip-audit
