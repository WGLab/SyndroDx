#!/bin/bash
#SBATCH --job-name=cranio-automatic-follow-on
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/logs/workflow-%j.private.log
set -uo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
trap '/opt/site/slurm/bin/scontrol requeue "$SLURM_JOB_ID"' USR1
while true; do
 /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/disease-diagnosis-model/gmdb_supervised_transface_20261026/workflow.py &
 child=$!
 wait "$child"
 code=$?
 if [ "$code" -eq 0 ]; then exit 0; fi
 sleep 60
done
