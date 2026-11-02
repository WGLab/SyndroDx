#!/bin/bash
#SBATCH --job-name=cf-pckpt
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/pckpt_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
CKPT="$1"; TAG=$(basename "$CKPT" .pt)
PY=$HOME/miniconda3/envs/qwen-finetune/bin/python
$PY -u check_collapse.py --ckpt "$CKPT"
$PY -u extract_ckpt_features.py --ckpt "$CKPT" --out "features_${TAG}.npy"
$PY -u probe_audit.py --features "features_${TAG}.npy" --skip-audit
