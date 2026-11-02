"""Offline-only frozen GMDB evaluation, separate from the active trainer."""
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
TRAIN = PROJECT / 'cranio_only_cpt_20261025_converge20'
V1 = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
BASE = V1.parent / 'dinov3_craniofacial_v2_official'
VENDOR = PROJECT / 'training_diagnostics_20261017/vendor/dinov3_fino'
GMDB = Path('/controlled-data/GMDB_v1.1.0')
METADATA = GMDB / 'gmdb_metadata_v1.1.0/gmdb_metadata'
CROPS = GMDB / 'gmdb_crops_v1.1.0/gmdb_crops'
sys.path[:0] = [str(WORK), str(VENDOR), str(BASE), str(V1)]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true', WANDB_MODE='disabled')
from common import load, save, digest, now, PYTHON

TABLES = ('gmdb_train_images', 'gmdb_val_images', 'gmdb_frequent_gallery_images',
          'gmdb_frequent_test_images', 'gmdb_rare_gallery_images', 'gmdb_rare_test_images')

def split_path(name):
    return METADATA / (name + '_v1.1.0.csv')
