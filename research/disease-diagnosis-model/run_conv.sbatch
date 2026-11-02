#!/bin/bash
#SBATCH --job-name=cf-conv
#SBATCH --partition=gpu-xe9680q
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/conv_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
$HOME/miniconda3/envs/qwen-finetune/bin/python -u probe_convergence.py
