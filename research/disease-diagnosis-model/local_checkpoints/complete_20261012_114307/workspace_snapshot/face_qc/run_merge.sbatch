#!/bin/bash
#SBATCH --job-name=faceqc-merge
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/merge_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/faceqc-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv-faceqc/bin/python -u face_qc/batch.py merge
