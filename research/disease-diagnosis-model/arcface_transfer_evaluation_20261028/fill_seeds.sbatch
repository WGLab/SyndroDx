#!/bin/bash
#SBATCH --job-name=arcface-fill-seeds
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/arcface_transfer_evaluation_20261028/logs/fill-seeds-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/arcface_transfer_evaluation_20261028/fill_seeds.py
