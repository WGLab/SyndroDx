#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/training/diagnostics/method_audit_v1/complete_audit_notice.py
