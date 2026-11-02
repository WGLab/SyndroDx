#!/bin/bash
set -euo pipefail
umask 077
exec /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official/export_encoder.py
