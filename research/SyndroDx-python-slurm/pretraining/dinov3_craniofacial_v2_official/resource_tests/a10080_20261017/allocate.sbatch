#!/bin/bash
set -euo pipefail
umask 077
/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official/resource_tests/a10080_20261017/public_benchmark.py
while true; do sleep 60; done
