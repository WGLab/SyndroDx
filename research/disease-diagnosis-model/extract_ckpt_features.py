#!/usr/bin/env python
"""Extract features from a DINO checkpoint's teacher backbone."""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--base', default='facebook/dinov2-base')
ap.add_argument('--which', default='teacher_backbone')
a, rest = ap.parse_known_args()
from transformers import AutoModel
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
m = AutoModel.from_pretrained(a.base)
m.load_state_dict(ck[a.which], strict=True)
tmp = os.path.join(HERE, '_ckpt_hf_tmp')
m.save_pretrained(tmp, safe_serialization=True)   # plain save, no accelerate path
print(f"epoch {ck['epoch']} -> {tmp}", flush=True)
os.execv(sys.executable, [sys.executable, os.path.join(HERE, 'extract_features.py'),
                          '--model', tmp, '--out', a.out, '--batch', '256', '--workers', '2'])
