#!/bin/bash
#SBATCH --job-name=faceqc-s4
#SBATCH --partition=defq
#SBATCH --array=0-31%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/screen_%A_%a.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/faceqc-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv-faceqc/bin/python -u face_qc/batch.py run --index "$SLURM_ARRAY_TASK_ID"
