#!/bin/bash
#SBATCH --job-name=gm-r50-public-download
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gestaltmatcher_arc_comparison_20261027/logs/r50-download-%j.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gestaltmatcher_arc_comparison_20261027/acquire_r50.py
