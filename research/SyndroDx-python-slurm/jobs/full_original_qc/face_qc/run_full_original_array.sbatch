#!/bin/bash
#SBATCH --job-name=orig-qc-all
#SBATCH --partition=defq
#SBATCH --array=0-63%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/original_%A_%a.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/original-qc-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv-faceqc/bin/python -u face_qc/full_audit.py run --index "$SLURM_ARRAY_TASK_ID"
