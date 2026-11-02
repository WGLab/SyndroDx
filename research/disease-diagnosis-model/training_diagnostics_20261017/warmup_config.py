"""Bounded frozen-backbone head warmup using pinned official FINO controls."""
import math
import sys
from pathlib import Path
WORK = Path(__file__).resolve().parent
VENDOR = WORK / 'vendor/dinov3_fino'
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
sys.path.insert(0, str(VENDOR))
from omegaconf import OmegaConf
from dinov3.configs import get_default_config


def configuration(world, phase):
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(BASE / 'training/config.yaml'))
    cfg.train.batch_size_per_gpu = 64
    cfg.train.OFFICIAL_EPOCH_LENGTH = 1
    cfg.train.num_workers = 4
    cfg.train.seed = 20261017
    cfg.train.output_dir = str(WORK / 'warmup_pilot' / phase)
    cfg.optim.epochs = 4 if phase == 'capacity' else 1000
    cfg.student.resume_from_teacher_chkpt = str(BASE / 'initialization/teacher_dcp')
    cfg.train.dataset_path = 'released_local_manifests'
    # The public FINO default config also serves fluorescence microscopy.
    # Explicitly select the RGB photographic branch for this experiment.
    cfg.train.cell_augmentation_type = None
    cfg.crops.vertical_flips = False
    cfg.crops.gaussian_blur = True
    cfg.train.compile = False
    cfg.student.fp8_enabled = False
    cfg.sigreg.enabled = False
    cfg.guide.enabled = False
    lr = cfg.schedules.lr
    scale = 4 * math.sqrt(64 * world / 1024)
    lr.start, lr.peak, lr.end = 0., 5e-5 / scale, 5e-5 / scale
    lr.warmup_iterations = 1 if phase == 'capacity' else 100
    lr.cosine_iterations = 0
    lr.freeze_last_layer_iterations = 1 if phase == 'capacity' else 16
    # This experiment ends before any backbone unfreezing is permitted.
    lr.freeze_student_iterations = 1000
    lr.backbone_warmup_after_freeze = 1000
    lr.freeze_patch_embed = True
    lr.freeze_cls_token = True
    lr.freeze_mask_token = True
    lr.unfreeze_last_n_blocks = 0
    if phase == 'capacity':
        cfg.schedules.teacher_temp.warmup_epochs = 1
    cfg.checkpointing.period = 4 if phase == 'capacity' else 100
    cfg.checkpointing.max_to_keep = 2
    cfg.checkpointing.keep_every = 10000
    cfg.evaluation.eval_period_iterations = 0
    return cfg


if __name__ == '__main__':
    import torch
    from unittest.mock import patch
    from dinov3.train import train
    cfg = configuration(12, 'training')
    with patch.object(train.distributed, 'get_world_size', return_value=12):
        schedule = train.build_schedulers(cfg)[0]
    assert len(schedule) == 1000 and abs(max(schedule) - 5e-5) < 1e-10
    with torch.device('meta'):
        from dinov3.train.ssl_meta_arch import SSLMetaArch
        model = SSLMetaArch(cfg)
    assert model.is_student_frozen(0) and model.is_student_frozen(999)
    from PIL import Image
    augmentation = model.build_data_augmentation_dino(cfg)
    for seed in range(4):
        torch.manual_seed(seed)
        views = augmentation(Image.new('RGB', (192, 224), (120, 90, 100)))
        assert len(views['global_crops']) == 2 and len(views['local_crops']) == 8
        assert all(v.shape == (3, 256, 256) and bool(torch.isfinite(v).all()) for v in views['global_crops'])
    print('Pinned FINO configuration, schedule and meta-device architecture checks passed.')
