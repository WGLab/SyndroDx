#!/bin/bash
#SBATCH --job-name=cf-dino
#SBATCH --partition=gpu-xe9680q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/logs/dino_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))
nvidia-smi --query-gpu=index,name,memory.total --format=csv
$HOME/miniconda3/envs/qwen-finetune/bin/torchrun \
  --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  ssl_train.py \
    --epochs 100 \
    --batch-size 64 \
    --workers 11 \
    --session-pair-prob 0.5 \
    --save-every 2 \
    --out-dir /opt/craniofacial/disease-diagnosis-model/ssl_ckpt
