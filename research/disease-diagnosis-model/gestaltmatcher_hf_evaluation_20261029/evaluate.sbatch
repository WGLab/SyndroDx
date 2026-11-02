#!/bin/bash
#SBATCH --job-name=gmdb-HF-R50-evaluation
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gestaltmatcher_hf_evaluation_20261029/logs/run-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gestaltmatcher_hf_evaluation_20261029/evaluate.py
