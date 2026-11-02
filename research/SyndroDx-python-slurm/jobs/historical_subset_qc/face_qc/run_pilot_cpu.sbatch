#!/bin/bash
#SBATCH --job-name=face-qc-pilot
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/pilot_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/faceqc-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv-faceqc/bin/python -u face_qc/pilot.py --n 300 --out face_qc/pilot300
