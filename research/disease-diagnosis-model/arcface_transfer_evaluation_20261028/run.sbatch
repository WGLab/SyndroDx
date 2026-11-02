#!/bin/bash
#SBATCH --job-name=arcface-transfer-evaluation
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/arcface_transfer_evaluation_20261028/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/arcface_transfer_evaluation_20261028/run.py
