#!/bin/bash
#SBATCH --job-name=gm-arc-matched-comparison
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gestaltmatcher_arc_comparison_20261027/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gestaltmatcher_arc_comparison_20261027/controller.py
