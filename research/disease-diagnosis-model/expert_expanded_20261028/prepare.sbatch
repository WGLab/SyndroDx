#!/bin/bash
#SBATCH --job-name=expert-expanded-audit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/expert_expanded_20261028/logs/prepare-%j.private.log
set -euo pipefail
umask 077
exec /usr/bin/python /opt/craniofacial/disease-diagnosis-model/expert_expanded_20261028/prepare.py
