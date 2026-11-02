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

ARMS = ('cranio_seed19', 'cranio_seed20')
SEEDS = {'cranio_seed19': 20261019, 'cranio_seed20': 20261020}
WORLD = 8
BATCH = 64
CAPACITY_STEPS = 16
STEPS = 1000
RESUME_STEP = 200
SOURCE_WORK = WORK.with_name('cranio_only_cpt_20261021')
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
    return dict(
        arms=list(ARMS), seeds=SEEDS, world_size=WORLD, batch_size_per_gpu=BATCH,
        global_batch_size=WORLD*BATCH, maximum_backbone_phase_steps=STEPS,
        resumed_from_completed_step=RESUME_STEP, additional_updates=STEPS-RESUME_STEP,
        original_backbone_initialization_before_checkpoint_restore=True,
        restore_student_teacher_optimizer_and_iteration=True,
        optimizer_reset_on_resume=False,
        sampler_resume='original seed and 200 local batches advanced; no sample-index replay',
        augmentation_resume='reseeded because worker RNG was not present in step-200 checkpoint; not bitwise identical to an uninterrupted run',
        validation_every_steps=PERIOD, head_peak_lr=HEAD_LR,
        maximum_backbone_lr=HEAD_LR*BACKBONE_LR_RATIO,
        backbone_lr_ramp_steps=100, initial_backbone_frozen_steps=1,
        differing_factor='random seed only; both repetitions use 100 percent released clinical craniofacial data',
        warmed_head_preparation_steps=0,
        warmed_head_preparation_is_additional_training_exposure=False,
        both_arms_same_original_backbone_ancestry=True, both_arms_fresh_optimizer=False,
        both_arms_same_data_and_augmentation_configuration=True,
        both_arms_same_random_seed=False,
        teacher_temperature=.07, teacher_momentum=.994,
        official_fino_commit='81aae9ad87be9a5e19f5043ac38957449af44b15',
        optimizer_loop_unmodified=True,
        local_adapters=['clinical-only approved training manifest',
                        'declared backbone LR multiplier', 'blocking checkpoint validation'],
        stopping='invalid metrics, centered CLS energy below half original, or decline >10 percentage points stop immediately; decline >2 percentage points with paired 95 percent CI upper bound <0 stops; smaller statistically supported decline at two consecutive checkpoints stops; other declines warn within the fixed 1000-step budget',
        guarded_metrics=['disease retrieval macro Top-1', 'disease ridge macro Top-1',
                         'structural retrieval macro Top-1', 'structural ridge macro Top-1'],
        ridge_selection='fixed original-encoder training-CV alpha: disease 100, structural 1000; never retuned per checkpoint',
        missing_nonfinite_or_failed_validation='stop before further optimizer updates',
        automatic_regression_restart=False, automatic_scale_up=False,
        arm_regression='stop this arm, preserve checkpoint, evaluate independent other arm',
        execution_or_integrity_failure='stop entire comparison',
        comparison='seed19 clinical-only versus preserved seed19 80/20 reference mixture at common steps with the same frozen probe; seed20 is an independent clinical-only repetition, not a paired mixture control',
        clinical_exposure='100 percent clinical at 512 samples per step; historical 80/20 trial had 20 percent less clinical exposure at equal optimizer steps; mixture attribution remains exploratory',
        resource_revision='2026-10-21: user requested continuation after two 200-step repetitions; extend to 1000 total steps',
        resource_selection='two fixed disjoint eight-H100 pools, one node per arm',
        total_gpus=16, capacity_steps=CAPACITY_STEPS, workers_per_rank=2,
        host_memory_per_rank_gib=[68]+[88]*7,
        memory_controls='synchronized garbage collection every 10 steps and at checkpoint/validation boundaries; host cgroup and GPU telemetry',
        input_verification='per-rank per-step hashes retained; different seeds are not expected to match',
        baseline='immutable original-encoder baseline; checkpoint-200 model, optimizer and saved validation features verified before step201',
        reference_face_fraction=0., reference_manifest_read=False,
        previous_mixed_trials_preserved=True,
        capacity='reuse passed 16-step eight-rank tests and real 50-step run on identical allocations, architecture and batch; initialization and memory checked again during this run',
        guard_tolerance_is_development_heuristic_not_clinical_noninferiority_margin=True,
        interpretation='exploratory development validation; not noninferiority or clinical validation',
        test_used=False, mapping_read=False, aggregate_chat_only=True,
    )
