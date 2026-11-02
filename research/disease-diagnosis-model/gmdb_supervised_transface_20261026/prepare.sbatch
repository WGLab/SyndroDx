#!/bin/bash
#SBATCH --job-name=gmdb-follow-on-prepare
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/logs/prepare-%j.private.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/prepare.py
