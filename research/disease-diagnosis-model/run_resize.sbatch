#!/bin/bash
#SBATCH --job-name=cf-resize
#SBATCH --partition=defq
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/resize_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
$HOME/miniconda3/envs/qwen-finetune/bin/python -u prep_resize.py
