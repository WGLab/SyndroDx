"""Predeclared, bounded head-initialization comparison; no clinical I/O."""
import math
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
DIAGNOSTICS = WORK.parent / 'training_diagnostics_20261017'
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
VENDOR = DIAGNOSTICS / 'vendor/dinov3_fino'
sys.path[:0] = [str(WORK), str(VENDOR), str(BASE), str(V1), str(DIAGNOSTICS)]

import json
ATTEMPT_PATH = os.environ.get('POOL_ATTEMPT_PLAN')
ATTEMPT = json.loads(Path(ATTEMPT_PATH).read_text()) if ATTEMPT_PATH else {}
ARMS = (ATTEMPT.get('arm', 'cranio_resume20'),)
SEEDS = {ARMS[0]: 20261019}
WORLD = ATTEMPT.get('world_size', 20)
BATCH = 32
CAPACITY_STEPS = 16
STEPS = 5000
import json
PLAN = json.loads((WORK/'resume_plan.json').read_text()) if (WORK/'resume_plan.json').exists() else dict(completed_steps=575)
RESUME_STEP = ATTEMPT.get('completed_steps', PLAN['completed_steps'])
SOURCE_TRAINING = Path(ATTEMPT.get('source_training', str(WORK/'source_snapshot')))
SOURCE_WORLD = ATTEMPT.get('source_world', 16)
CONSUMED_SAMPLES = ATTEMPT.get('consumed_global_samples', 575*512)
PROBE_FOLDER = WORK/ARMS[0]/'gpu_probe' if ATTEMPT_PATH else WORK/'gpu_probe'
RESIZE_REQUEST = WORK/ARMS[0]/'resize_requested.json'
PERIOD = 25
SEED = 20261019
HEAD_LR = 5e-5
BACKBONE_LR_RATIO = .02  # Maximum last-block LR 1e-6, before layerwise decay.
WARMED_CHECKPOINT = DIAGNOSTICS / 'warmup_pilot/training/ckpt/999'


def configuration(arm, phase='training'):
    from warmup_config import configuration as previous_configuration
    cfg = previous_configuration(WORLD, 'training')
    cfg.train.output_dir = str(WORK / arm / phase)
    cfg.train.seed = SEEDS[arm]
    cfg.optim.epochs = CAPACITY_STEPS if phase == 'capacity' else STEPS
    cfg.train.batch_size_per_gpu = BATCH
    cfg.train.num_workers = 2
    cfg.evaluation.eval_period_iterations = 0  # Gate after a FULL optimizer checkpoint.
    cfg.checkpointing.period = 4 if phase == 'capacity' else PERIOD
    cfg.checkpointing.max_to_keep = 3
    cfg.checkpointing.keep_every = 100000
    lr = cfg.schedules.lr
    scale = 4 * math.sqrt(BATCH * WORLD / 1024)
    lr.start, lr.peak, lr.end = 0., HEAD_LR / scale, HEAD_LR / scale
    lr.warmup_iterations = 1 if phase == 'capacity' else 25
    lr.cosine_iterations = 0
    lr.freeze_last_layer_iterations = 1 if phase == 'capacity' else 16
    # Identical one-step freeze and 100-step backbone ramp in BOTH arms.
    lr.freeze_student_iterations = 1
    lr.backbone_warmup_after_freeze = 2 if phase == 'capacity' else 100
    lr.freeze_patch_embed = lr.freeze_cls_token = lr.freeze_mask_token = True
    lr.unfreeze_last_n_blocks = 0
    for name, value in [('teacher_temp', .07), ('momentum', .994), ('weight_decay', .04)]:
        schedule = getattr(cfg.schedules, name)
        schedule.start = schedule.peak = schedule.end = value
        schedule.warmup_iterations = 0
        schedule.cosine_iterations = 0
    return cfg


def protocol():
    return dict(world_size=WORLD, batch_size_per_gpu=BATCH,global_batch_size=WORLD*BATCH,
        resumed_completed_steps=RESUME_STEP,target_steps=STEPS,midpoint_steps=1000,
        source_world_size=SOURCE_WORLD,consumed_global_samples_at_resume=CONSUMED_SAMPLES,
        model='DINOv3 ViT-7B',independent_models=1,clinical_only=True,reference_fraction=0.,
        head_learning_rate=HEAD_LR,maximum_backbone_learning_rate=HEAD_LR*BACKBONE_LR_RATIO,
        optimizer_reset=False,losses_unchanged=True,validation_every=PERIOD,
        fixed_probe_alpha={'disease':100,'structural':1000},
        explicit_resume_after_step575_guard=True,
        original_baseline_metrics_always_reported=True,
        new_guard_anchor='step575; user explicitly authorized continuing after its documented regression',
        guard='same thresholds against step575; nonfinite loss, failed validation, original-relative feature collapse and no-progress guards retained',
        sampler='original 16 rank-specific shuffled lanes interleaved into a canonical stream, advanced by cumulative exposure and repartitioned across current ranks',
        augmentation_rng='reseeded at resize; not bitwise equivalent to uninterrupted fixed-world training',
        original_validation_reports_immutable=True,final_test_used=False,mapping_read=False,
        trackers_disabled=True,patient_data_remains_local=True,
        resource_policy='twenty separate one-H100 batch leases; start with at least four GPUs; expand at validated full checkpoints; useful feature-diagnostic work on additional GPUs; one distributed model; leases exit after training finishes or fails')
