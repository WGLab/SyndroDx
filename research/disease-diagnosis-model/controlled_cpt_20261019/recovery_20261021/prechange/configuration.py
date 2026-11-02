"""Predeclared, bounded head-initialization comparison; no clinical I/O."""
import math
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
DIAGNOSTICS = WORK.parent / 'training_diagnostics_20261017'
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
VENDOR = DIAGNOSTICS / 'vendor/dinov3_fino'
sys.path[:0] = [str(VENDOR), str(BASE), str(V1), str(DIAGNOSTICS)]

ARMS = ('warmed_heads', 'random_heads')
WORLD = 8
BATCH = 64
STEPS = 200
PERIOD = 25
SEED = 20261019
HEAD_LR = 5e-5
BACKBONE_LR_RATIO = .02  # Maximum last-block LR 1e-6, before layerwise decay.
WARMED_CHECKPOINT = DIAGNOSTICS / 'warmup_pilot/training/ckpt/999'


def configuration(arm, phase='training'):
    from warmup_config import configuration as previous_configuration
    cfg = previous_configuration(WORLD, 'training')
    cfg.train.output_dir = str(WORK / arm / phase)
    cfg.train.seed = SEED
    cfg.optim.epochs = 4 if phase == 'capacity' else STEPS
    cfg.train.batch_size_per_gpu = BATCH
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
    return dict(
        arms=list(ARMS), world_size=WORLD, batch_size_per_gpu=BATCH,
        global_batch_size=WORLD*BATCH, maximum_backbone_phase_steps=STEPS,
        validation_every_steps=PERIOD, head_peak_lr=HEAD_LR,
        maximum_backbone_lr=HEAD_LR*BACKBONE_LR_RATIO,
        backbone_lr_ramp_steps=100, initial_backbone_frozen_steps=1,
        differing_factor='student and EMA DINO/iBOT head initialization only',
        warmed_head_preparation_steps=1000,
        warmed_head_preparation_is_additional_training_exposure=True,
        both_arms_original_backbone=True, both_arms_fresh_optimizer=True,
        both_arms_same_data_seed_and_augmentations=True,
        teacher_temperature=.07, teacher_momentum=.994, seed=SEED,
        official_fino_commit='81aae9ad87be9a5e19f5043ac38957449af44b15',
        optimizer_loop_unmodified=True,
        local_adapters=['approved local dataset', 'head-only initialization loading',
                        'declared backbone LR multiplier', 'blocking checkpoint validation'],
        stopping='any guarded Top-1 below original OR last accepted checkpoint; no patience',
        guarded_metrics=['disease retrieval macro Top-1', 'disease ridge macro Top-1',
                         'structural retrieval macro Top-1', 'structural ridge macro Top-1'],
        ridge_selection='five-fold training patients only; same folds and alpha grid',
        missing_nonfinite_or_failed_validation='stop before further optimizer updates',
        automatic_regression_restart=False, automatic_scale_up=False,
        arm_regression='stop this arm, preserve checkpoint, evaluate independent other arm',
        execution_or_integrity_failure='stop entire comparison',
        comparison='common completed checkpoints only; early stopping reported',
        interpretation='exploratory development validation; not noninferiority or clinical validation',
        test_used=False, mapping_read=False, aggregate_chat_only=True,
    )
