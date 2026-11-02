#!/bin/bash
#SBATCH --job-name=face-vlm-pilot
#SBATCH --partition=gpu-xe9680q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/vlm_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
/opt/craniofacial/user_home/miniconda3/envs/qwen-finetune/bin/python -u face_qc/vlm_screen.py
