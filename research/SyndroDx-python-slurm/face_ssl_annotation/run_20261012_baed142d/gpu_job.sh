#!/bin/bash
#SBATCH --job-name=qwen32b-ssl-face
#SBATCH --partition=gpu-xe9680q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=448G
#SBATCH --time=8-00:00:00
#SBATCH --output=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/gpu-%j.log
#SBATCH --error=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/gpu-%j.err
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

export LOCAL_QWEN_PYTHON=/opt/craniofacial/user_home/miniconda3/envs/qwen-finetune/bin/python
export LOCAL_QWEN_RUN=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d
cd /opt/craniofacial/SyndroDx-python-slurm
srun --ntasks=8 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --output=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/worker-%t-%j.log --error=/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/logs/worker-%t-%j.err bash --noprofile --norc -c '"$LOCAL_QWEN_PYTHON" "$LOCAL_QWEN_RUN/annotation_pipeline.py" worker --worker "$SLURM_PROCID"'
/opt/craniofacial/user_home/miniconda3/envs/qwen-finetune/bin/python /opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d/annotation_pipeline.py finalize
