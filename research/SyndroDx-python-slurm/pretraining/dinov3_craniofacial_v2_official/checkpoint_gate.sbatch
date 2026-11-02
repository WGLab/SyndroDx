#!/bin/bash
set -euo pipefail
umask 077
exec /usr/bin/python3 /opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official/checkpoint_batch_gate.py
