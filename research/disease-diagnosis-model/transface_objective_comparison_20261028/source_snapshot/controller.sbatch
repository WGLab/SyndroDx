#!/bin/bash
#SBATCH --job-name=transface-arcface-vs-ssl
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/transface_objective_comparison_20261028/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/transface_objective_comparison_20261028/controller.py
