"""Blocking local validation. Images, identifiers and features stay on HPC."""
import math
import random
import sys
from contextlib import contextmanager

from configuration import WORK, BASE, V1, DIAGNOSTICS
sys.path[:0] = [str(BASE), str(V1), str(DIAGNOSTICS)]
import numpy as np
import torch
import torch.distributed as dist
from common import load, save, digest, now
from validation_hook import cohort
from evaluation_worker import fit_image
from evaluation_core import aggregate_patients, retrieval, macro_score, paired_interval
from frozen_probe import ridge_predictions, label_hits
from guard import decide, KEYS, EPSILON
from run_policy import validation_policy

CATEGORIES = [('disease_or_disorder', 'disease'), ('structural_diagnosis', 'structural')]


@contextmanager
def preserve_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


def scores(records, features, labels):
    partitions = {split: aggregate_patients(records, features, split) for split in ('train', 'validation')}
    gids, g = partitions['train']; qids, q = partitions['validation']
    if set(gids) & set(qids):
        raise RuntimeError('patient_split_overlap')
    metrics, details, private_hits = {}, {}, {}
    for category, short in CATEGORIES:
        lab = labels[category]
        gi = [i for i, key in enumerate(gids) if lab.get(key)]
        qi = [i for i, key in enumerate(qids) if lab.get(key)]
        gy = [set(lab[gids[i]]) for i in gi]
        qy = [set(lab[qids[i]]) for i in qi]
        if not gi or not qi:
            raise RuntimeError('required_endpoint_unavailable')
        hits, coverage = retrieval(q[qi], g[gi], [qids[i] for i in qi], [gids[i] for i in gi], qy, gy)
        alpha = load(WORK/'fixed_probe_parameters.json')['alpha'][short]
        cv = dict(selected_alpha=alpha, fixed_before_training=True,
                  alpha_source='original encoder training-patient cross-validation')
        vocabulary = sorted(set().union(*gy))
        predictions = ridge_predictions(g[gi], q[qi], gy, vocabulary, [alpha])[alpha]
        rhits = {k: label_hits(predictions, qy, vocabulary, k) for k in (1, 5, 10)}
        metrics[short+'_retrieval_top1'] = macro_score(hits[1], qy)
        metrics[short+'_ridge_top1'] = macro_score(rhits[1], qy)
        details[short] = dict(training_patients=len(gi), validation_patients=len(qi),
                             training_classes=len(vocabulary), gallery_coverage=float(coverage.mean()),
                             retrieval={f'top{k}_macro': macro_score(hits[k], qy) for k in (1, 5, 10)},
                             ridge=dict(cv, **{f'top{k}_macro': macro_score(rhits[k], qy) for k in (1, 5, 10)}))
        private_hits[short] = (hits[1], rhits[1], qy)
    return metrics, details, private_hits


def rank_zero_report(folder, step, records, current, labels):
    baseline_path = WORK/'original_runtime_baseline.private.npy'
    baseline_report_path = WORK/'original_runtime_baseline.json'
    metrics, details, hits = scores(records, current, labels)
    if step == 0 and not baseline_path.exists():
        # A fresh inference through the exact same pinned runtime is the anchor.
        np.save(baseline_path, current)
        save(baseline_report_path, dict(state='complete', metrics=metrics, endpoints=details,
             features_sha256=digest(baseline_path), test_used=False,
             validation_inputs_sha256=digest(BASE/'validation/inputs.private.jsonl'),
             created_at_utc=now()))
        for short, (kh, rh, _) in hits.items():
            np.savez(WORK/(short+'_original_hits.private.npz'), retrieval=kh, ridge=rh)
    baseline_report = load(baseline_report_path)
    if baseline_report.get('state') != 'complete' or digest(baseline_path) != baseline_report.get('features_sha256'):
        raise RuntimeError('baseline_missing_or_modified')
    baseline = np.load(baseline_path, allow_pickle=False)
    original = baseline_report['metrics']
    if step == 0:
        cosine = (baseline*current).sum(1)
        if min(cosine) < .99999 or any(abs(metrics[k]-original[k]) > EPSILON for k in KEYS):
            raise RuntimeError('paired_original_encoder_baseline_mismatch')
    previous_receipt = load(folder.parent/'last_accepted.json')
    previous = previous_receipt.get('metrics', original)
    comparisons = {}
    for short, (kh, rh, terms) in hits.items():
        with np.load(WORK/(short+'_original_hits.private.npz'), allow_pickle=False) as reference:
            comparisons[short] = dict(
                retrieval=paired_interval(reference['retrieval'], kh, terms, repetitions=1000),
                ridge=paired_interval(reference['ridge'], rh, terms, repetitions=1000))
    monitor_state = load(folder.parent/'monitor_state.json')
    anchor=load(WORK/'resume_anchor.json')
    if anchor.get('state')!='passed' or anchor.get('completed_steps')!=575:
        raise RuntimeError('explicit_resume_anchor_required')
    resume_comparisons={}
    for short,(kh,rh,terms) in hits.items():
        with np.load(WORK/(short+'_resume_hits.private.npz'),allow_pickle=False) as reference:
            resume_comparisons[short]=dict(
                retrieval=paired_interval(reference['retrieval'],kh,terms,repetitions=1000),
                ridge=paired_interval(reference['ridge'],rh,terms,repetitions=1000))
    decision = decide(metrics, anchor['metrics'], previous, comparisons=resume_comparisons,
                      previous_supported=monitor_state.get('supported_declines', []))
    decision = validation_policy(decision)
    decision['guard_anchor_completed_steps']=575
    decision['explicit_resume_after_documented_original_relative_regression']=True
    decision['original_relative_declines']=[dict(metric=k,delta=metrics[k]-original[k]) for k in KEYS if metrics[k]<original[k]-EPSILON]
    baseline_energy = float(np.mean(np.sum((baseline-baseline.mean(0))**2, axis=1)))
    current_energy = float(np.mean(np.sum((current-current.mean(0))**2, axis=1)))
    if not math.isfinite(current_energy) or current_energy < .5*baseline_energy:
        decision = dict(continue_training=False, state='paused_feature_collapse',
                        supported_declines=[], reason='centered_CLS_energy_below_half_original')
    historical = np.load(BASE/'validation/original_baseline.private.npy', mmap_mode='r', allow_pickle=False)
    historical_cosine = (historical*current).sum(1) if step == 0 else None
    report = dict(state='complete', completed_steps=step, metrics=metrics, endpoints=details,
                  original_metrics=original, previous_accepted_metrics=previous, decision=decision,
                  paired_comparison=comparisons, paired_resume_comparison=resume_comparisons, test_used=False,
                  fixed_probe_parameters=load(WORK/'fixed_probe_parameters.json')['alpha'],
                  centered_CLS_energy=dict(original=baseline_energy, current=current_energy),
                  metadata_labels_not_clinical_gold_standard=True,
                  exploratory_repeated_validation_not_noninferiority_proof=True,
                  original_historical_runtime_min_cosine=float(min(historical_cosine)) if step == 0 else None,
                  finished_at_utc=now())
    save(folder/'report.json', report)
    save(folder.parent/'monitor_state.json',dict(completed_steps=step,
         supported_declines=decision.get('supported_declines', [])))
    if decision['state'] == 'paused_invalid_validation':
        raise RuntimeError('invalid_validation_metrics')
    if decision['continue_training']:
        save(folder.parent/'last_accepted.json', dict(completed_steps=step, metrics=metrics,
                                                     report_path=str(folder/'report.json')))
    save(folder.parent.parent/'latest_validation.json', dict(report_path=str(folder/'report.json'),
         completed_steps=step, metrics=metrics, decision=decision, updated_at_utc=now()))
    return decision


def validate(cfg, model, step, output):
    rank, world = dist.get_rank(), dist.get_world_size()
    folder = output/'validation'/f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    with preserve_rng():
        records, _, labels = cohort()
        train_ids = {r['group_id'] for r in records if r['split'] == 'train'}
        query_ids = {r['group_id'] for r in records if r['split'] == 'validation'}
        if train_ids & query_ids:
            raise RuntimeError('patient_overlap')
        indices = list(range(rank, len(records), world))
        batch = 16
        outputs = np.empty((len(indices), model.embed_dim), dtype=np.float32)
        for start in range(0, math.ceil(len(records)/world), batch):
            selected = indices[start:start+batch]
            valid = torch.ones((), device='cuda', dtype=torch.int)
            tensors = []
            try:
                for i in selected:
                    if digest(records[i]['crop_path']) != records[i]['crop_sha256']:
                        raise RuntimeError('released_crop_modified')
                    tensors.append(fit_image(records[i]['crop_path'], cfg))
                while len(tensors) < batch:
                    tensors.append(torch.zeros(3, cfg.crops.global_crops_size, cfg.crops.global_crops_size))
                inputs = torch.stack(tensors).cuda(non_blocking=True)
            except Exception:
                valid.zero_()
                inputs = torch.zeros(batch, 3, cfg.crops.global_crops_size, cfg.crops.global_crops_size, device='cuda')
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            if not bool(valid):
                raise RuntimeError('validation_input_failed')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                value = model.teacher.backbone(inputs, is_training=True)['x_norm_clstoken'].float()
                value = torch.nn.functional.normalize(value, dim=-1)
            finite = torch.isfinite(value).all().to(torch.int)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise RuntimeError('validation_nonfinite_features')
            if selected:
                outputs[start:start+len(selected)] = value[:len(selected)].cpu().numpy()
        np.save(folder/f'rank_{rank:02d}.private.npy', outputs)
        dist.barrier()
        status = torch.zeros((), device='cuda', dtype=torch.int)
        if rank == 0:
            try:
                current = np.empty((len(records), model.embed_dim), dtype=np.float32)
                for i in range(world):
                    shard = np.load(folder/f'rank_{i:02d}.private.npy', allow_pickle=False)
                    if shard.shape != current[i::world].shape:
                        raise RuntimeError('validation_shard_mismatch')
                    current[i::world] = shard
                decision = rank_zero_report(folder, step, records, current, labels)
                status.fill_(1 if decision['continue_training'] else 2)
            except Exception as error:
                save(folder/'failure.json', dict(state='failed', error_type=type(error).__name__,
                     test_used=False, updated_at_utc=now()))
        dist.broadcast(status, src=0)
        if int(status) == 0:
            raise RuntimeError('blocking_validation_failed')
        return int(status) == 1
