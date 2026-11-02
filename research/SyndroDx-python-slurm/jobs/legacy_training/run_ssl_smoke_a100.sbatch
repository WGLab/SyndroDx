#!/bin/bash
#SBATCH --job-name=cf-ssl-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/sslsmoke_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
$HOME/miniconda3/envs/qwen-finetune/bin/python -u ssl_train.py \
  --epochs 1 --limit-steps 60 --batch-size 32 --workers 2 --save-every 999 \
  --out-dir /opt/craniofacial/disease-diagnosis-model/ssl_ckpt_smoke
