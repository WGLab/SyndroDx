#!/bin/bash
#SBATCH --job-name=GMDB-disease-ArcFace-finetune
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_arcface_finetune_20261029/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gmdb_arcface_finetune_20261029/controller.py
