#!/bin/bash
#SBATCH --job-name=GMDB-ArcFace-then-WCE
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_arcface_then_wce_20261030/logs/controller-%j.private.log
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gmdb_arcface_then_wce_20261030/controller.py
