#!/bin/bash
#SBATCH --job-name=orig-qc-merge
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/face_qc/logs/original_merge_%j.out
set -euo pipefail
cd /opt/craniofacial/disease-diagnosis-model
.venv-faceqc/bin/python -u face_qc/full_audit.py merge
