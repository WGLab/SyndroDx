#!/bin/bash
#SBATCH --job-name=cf-collapse
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/collapse_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
$HOME/miniconda3/envs/qwen-finetune/bin/python -u check_collapse.py --ckpt "$1"
