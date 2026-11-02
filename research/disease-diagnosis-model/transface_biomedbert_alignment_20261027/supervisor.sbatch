#!/bin/bash
#SBATCH --job-name=cranio-HPO-alignment
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/transface_biomedbert_alignment_20261027/logs/supervisor-%j.private.log
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/transface_biomedbert_alignment_20261027/supervisor.py
