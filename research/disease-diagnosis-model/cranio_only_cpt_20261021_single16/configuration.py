"""Predeclared, bounded head-initialization comparison; no clinical I/O."""
import math
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
DIAGNOSTICS = WORK.parent / 'training_diagnostics_20261017'
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
VENDOR = DIAGNOSTICS / 'vendor/dinov3_fino'
sys.path[:0] = [str(WORK), str(VENDOR), str(BASE), str(V1), str(DIAGNOSTICS)]

ARMS = ('cranio_single16',)
SEEDS = {'cranio_single16': 20261019}
WORLD = 16
BATCH = 32
CAPACITY_STEPS = 16
STEPS = 1000
import json
PLAN = json.loads((WORK/'resume_plan.json').read_text()) if (WORK/'resume_plan.json').exists() else dict(completed_steps=300)
RESUME_STEP = PLAN['completed_steps']
SOURCE_TRAINING = WORK/'source_snapshot'
SOURCE_WORLD = 8
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
    return dict(arms=list(ARMS), world_size=WORLD, batch_size_per_gpu=BATCH,
        global_batch_size=WORLD*BATCH, seed=SEED, independent_repetitions=1,
        resumed_from_completed_step=RESUME_STEP, target_completed_steps=STEPS,
        source_world_size=SOURCE_WORLD, dcp_resharding='8 to 16 ranks',
        source_selection='fixed first repetition; latest complete checkpoint passing existing guard, not chosen by score',
        restore_student_teacher_optimizer_and_iteration=True, optimizer_reset=False,
        reference_face_fraction=0., reference_manifest_read=False, mapping_read=False,
        head_peak_lr=HEAD_LR, maximum_backbone_lr=HEAD_LR*BACKBONE_LR_RATIO,
        validation_every_steps=PERIOD, fixed_ridge_alphas={'disease':100,'structural':1000},
        test_used=False, aggregate_chat_only=True, offline=True,
        optimizer_loop_unmodified=True, target='single shared craniofacial encoder',
        automatic_regression_restart=False,
        effective_batch_and_schedules_preserved=True,
        not_bitwise_equivalent='rank assignment, local KoLeo neighborhood and worker augmentation RNG change',
        sampler_resume='same global index stream and total consumed sample offset; no restart from beginning',
        stopping='existing prospective validation and memory guards retained',
        topology='two nodes, eight GPUs per node; one global FSDP process group',
        resource_revision='user requested one model across all sixteen H100 GPUs')
