"""Current-checkpoint versus original backbone on validation only, entirely on HPC."""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg
from runtime import ROOT, load, save, digest, now, notify
from weights import OfficialBackbone
from evaluation_worker import fit_image, forward
from evaluation_core import normalized, retrieval, macro_score, collapse_statistics, paired_interval


def cohort(records, labels, maximum=256):
    grouped = collections.defaultdict(list)
    for row in records:
        if row['split'] == 'validation':
            grouped[row['group_id']].append(row)
    by_label = collections.defaultdict(list)
    for group in grouped:
        for label in labels.get(group, []):
            by_label[label].append(group)
    selected = set()
    # Up to eight independent validation patients/category; no test labels or
    # test images influence selection. Stable hashes define a diagnostic split.
    for label in sorted(by_label):
        if len(set(by_label[label])) < 2:
            continue
        for group in sorted(set(by_label[label]), key=lambda g: hashlib.sha256(g.encode()).hexdigest())[:8]:
            selected.add(group)
    if not selected:
        selected = set(grouped)
    selected = sorted(selected, key=lambda g: hashlib.sha256(g.encode()).hexdigest())[:maximum]
    result = []
    for group in selected:
        chosen = []
        for view in ['frontal', 'profile', 'oblique']:
            candidates = sorted([r for r in grouped[group] if r['view'] == view], key=lambda r: r['crop_sha256'])
            if candidates:
                chosen.append(candidates[0])
            if len(chosen) == 2:
                break
        result.extend(chosen)
    if any(row['split'] != 'validation' for row in result):
        raise RuntimeError('nonvalidation_image_in_diagnostic')
    return result


def load_model(cfg, checkpoint=None):
    with torch.device('meta'):
        model, _ = build_model_from_cfg(cfg, only_teacher=True)
    model.to_empty(device='cuda')
    model.rope_embed.dtype = torch.float32
    model.init_weights()
    if checkpoint is None:
        OfficialBackbone(ROOT / 'assets/dinov3-vit7b16').load(model)
    else:
        state = {'model.teacher.backbone.' + key: value for key, value in model.state_dict().items()}
        dcp.load(state, storage_reader=dcp.FileSystemReader(checkpoint), planner=DefaultLoadPlanner(allow_partial_load=True))
    return model.eval()


def main(expanded=False):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    folder = ROOT / 'training/diagnostics'
    folder.mkdir(exist_ok=True)
    cfg = OmegaConf.load(ROOT / 'training/run_config.yaml')
    checkpoints = [p for p in (ROOT / 'training/ckpt').iterdir() if p.is_dir() and p.name.isdigit() and (p / '.metadata').is_file()]
    checkpoint = max(checkpoints, key=lambda p: int(p.name))
    plan = load(ROOT / 'evaluation/input_plan.json')
    path = ROOT / 'evaluation/inputs.private.jsonl'
    if digest(path) != plan['inputs_sha256']:
        raise RuntimeError('validation_input_manifest_changed')
    with path.open() as source:
        records = [json.loads(line) for line in source]
    sources = load(ROOT / 'evaluation/labels.private.json')
    all_labels = collections.defaultdict(set)
    for groups in sources.get('labels', {}).values():
        for group, terms in groups.items():
            all_labels[group].update(terms)
    if expanded:
        grouped_rows = collections.defaultdict(list)
        for row in records:
            if row['split'] in ['train', 'validation'] and all_labels.get(row['group_id']):
                grouped_rows[row['group_id']].append(row)
        selected = []
        for group in sorted(grouped_rows):
            views = set()
            for row in sorted(grouped_rows[group], key=lambda r: (['frontal','profile','oblique'].index(r['view']), r['crop_sha256'])):
                if row['view'] not in views:
                    selected.append(row); views.add(row['view'])
                if len(views) == 2:break
        prefix = 'expanded_validation_audit'
    else:
        selected = cohort(records, all_labels)
        prefix = 'validation_audit'
    if not selected:
        raise RuntimeError('no_released_validation_images')
    outputs = {}
    summary = dict(state='running', checkpoint_iteration=int(checkpoint.name),
                   validation_queries_only=True, train_gallery=expanded, test_used=False, diagnostic_not_final_evaluation=True,
                   metadata_labels_not_clinically_adjudicated=True, models={})
    for kind in ['original_dinov3', 'current_teacher']:
        model = load_model(cfg, None if kind == 'original_dinov3' else checkpoint)
        features = []
        for start in range(0, len(selected), 16):
            batch = selected[start:start + 16]
            if any(digest(row['crop_path']) != row['crop_sha256'] for row in batch):
                raise RuntimeError('released_validation_image_changed')
            tensor = torch.stack([fit_image(row['crop_path'], cfg) for row in batch])
            features.append(forward(model, tensor))
            save(folder / (prefix + '_progress.json'), dict(state='extracting', model=kind, updated_at_utc=now()))
        values = np.concatenate(features)
        grouped = collections.defaultdict(list)
        for row, value in zip(selected, values):
            grouped[row['group_id']].append(value)
        ids = sorted(grouped)
        vectors = normalized([np.mean(grouped[g], axis=0) for g in ids])
        outputs[kind] = (ids, vectors)
        summary['models'][kind] = dict(collapse=collapse_statistics(vectors))
        del model, features, values
        gc.collect()
        torch.cuda.empty_cache()
    summary['metadata_label_endpoints'] = {}
    for category, labels in sources.get('labels', {}).items():
        endpoint, successes = {}, {}
        for kind, (ids, vectors) in outputs.items():
            eligible = [i for i, g in enumerate(ids) if labels.get(g)]
            if expanded:
                group_split = {row['group_id']: row['split'] for row in selected}
                gallery = [i for i in eligible if group_split[ids[i]] == 'train']
                query = [i for i in eligible if group_split[ids[i]] == 'validation']
            else:
                gallery = eligible[::2]
                query = eligible[1::2]
            if not gallery or not query:
                endpoint[kind] = dict(state='not_evaluable')
                continue
            y_query = [set(labels[ids[i]]) for i in query]
            hits, covered = retrieval(vectors[query], vectors[gallery], [ids[i] for i in query],
                                       [ids[i] for i in gallery], y_query, [set(labels[ids[i]]) for i in gallery])
            endpoint[kind] = dict(state='complete', top1_macro=macro_score(hits[1], y_query),
                                  top5_macro=macro_score(hits[5], y_query), gallery_label_coverage=float(covered.mean()))
            successes[kind] = (hits[1], y_query)
        if len(successes) == 2:
            first, terms = successes['original_dinov3']
            second, _ = successes['current_teacher']
            endpoint['paired_top1'] = paired_interval(first, second, terms)
        summary['metadata_label_endpoints'][category] = endpoint
    summary.update(state='complete', completed_at_utc=now(), raw_images_or_embeddings_exported=False)
    save(folder / (prefix + '.private.json'), summary)
    save(folder / (prefix + '_status.json'), dict(state='complete', checkpoint_iteration=int(checkpoint.name),
         validation_queries_only=True, train_gallery=expanded, test_used=False, completed_at_utc=now()))
    notify('Current encoder validation audit completed ' + prefix, 'Encoder validation audit completed',
           'A small offline validation-only comparison of the current checkpoint and original backbone is complete. This is a diagnostic, not the final held-out test or clinical validation. All images, identifiers, embeddings and results remain on HPC.')


if __name__ == '__main__':
    os.umask(0o077)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    try:
        main(args.expanded)
    except Exception as exc:
        save(ROOT / 'training/diagnostics' / ('expanded_validation_audit_status.json' if args.expanded else 'validation_audit_status.json'), dict(state='failed', error_type=type(exc).__name__, updated_at_utc=now()))
        raise SystemExit(1)
