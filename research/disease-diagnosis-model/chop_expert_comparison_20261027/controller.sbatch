#!/bin/bash
#SBATCH --job-name=chop-expert-four-models
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/chop_expert_comparison_20261027/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/chop_expert_comparison_20261027/controller.py
