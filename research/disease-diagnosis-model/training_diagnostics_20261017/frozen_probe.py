"""HPC-only frozen-feature diagnostic. Public code; private inputs never printed."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, os.environ.get("SLURM_CPUS_PER_TASK", "4"))
import collections
import json
import sys
from pathlib import Path
import numpy as np

BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
sys.path.insert(0, str(V1))
sys.path.insert(0, str(BASE))
from common import digest, load, save, now, notify
from evaluation_core import aggregate_patients, macro_score, paired_interval

ALPHAS = (.01, .1, 1., 10., 100., 1000., 10000., 100000., 1000000.)
SEED = 20261017
REPORT_STEM = os.environ.get('PROBE_REPORT_STEM', 'frozen_probe_extended_alpha')


def targets(terms, vocabulary):
    table = {name: i for i, name in enumerate(vocabulary)}
    result = np.zeros((len(terms), len(vocabulary)), dtype=np.float64)
    for row, names in enumerate(terms):
        for name in names:
            if name in table:
                result[row, table[name]] = 1.
    return result


def ridge_predictions(train, query, terms, vocabulary, alphas=ALPHAS):
    """Weighted ridge with unpenalized intercept, solved in patient space."""
    x, q = np.asarray(train, np.float64), np.asarray(query, np.float64)
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('nonfinite_input')
    y = targets(terms, vocabulary)
    counts = np.maximum(y.sum(0), 1.)
    weights = ((y / counts).sum(1) / np.maximum(y.sum(1), 1.))
    if np.any(weights <= 0):
        raise ValueError('training_patient_without_target')
    weights *= len(weights) / weights.sum()
    mean, scale = x.mean(0), np.maximum(x.std(0), .01)
    x, q = (x - mean) / scale, (q - mean) / scale
    xmean = np.average(x, axis=0, weights=weights)
    ymean = np.average(y, axis=0, weights=weights)
    a = (x - xmean) * np.sqrt(weights[:, None])
    b = (y - ymean) * np.sqrt(weights[:, None])
    eigenvalues, basis = np.linalg.eigh(a @ a.T)
    eigenvalues = np.maximum(eigenvalues, 0.)
    projected = ((q - xmean) @ a.T) @ basis
    rhs = basis.T @ b
    return {alpha: projected @ (rhs / (eigenvalues[:, None] + alpha)) + ymean
            for alpha in alphas}


def label_hits(scores, terms, vocabulary, k=1):
    indices = np.argsort(-scores, axis=1, kind='stable')[:, :min(k, len(vocabulary))]
    return np.asarray([bool(set(vocabulary[i] for i in row) & truth)
                       for row, truth in zip(indices, terms)], dtype=bool)


def training_folds(terms, count=5):
    """Deterministic patient folds; primary label used only for stratification."""
    groups = collections.defaultdict(list)
    for i, labels in enumerate(terms):
        groups[sorted(labels)[0]].append(i)
    rng = np.random.default_rng(SEED)
    fold = np.empty(len(terms), dtype=np.int64)
    for label in sorted(groups):
        indices = np.asarray(groups[label])
        rng.shuffle(indices)
        offset = int(rng.integers(count))
        fold[indices] = (np.arange(len(indices)) + offset) % count
    return fold


def select_alpha(train, terms, fold):
    """Only training patients are available to this selection function."""
    successes = {alpha: np.zeros(len(terms), dtype=bool) for alpha in ALPHAS}
    covered = np.zeros(len(terms), dtype=bool)
    for part in sorted(set(fold)):
        ti, vi = np.flatnonzero(fold != part), np.flatnonzero(fold == part)
        ty = [terms[i] for i in ti]
        vy = [terms[i] for i in vi]
        vocabulary = sorted(set().union(*ty))
        predictions = ridge_predictions(train[ti], train[vi], ty, vocabulary)
        covered[vi] = [bool(y & set(vocabulary)) for y in vy]
        for alpha, scores in predictions.items():
            successes[alpha][vi] = label_hits(scores, vy, vocabulary)
    values = {alpha: macro_score(hits, terms) for alpha, hits in successes.items()}
    best = max(ALPHAS, key=lambda alpha: (values[alpha], alpha))
    return best, {'selected_alpha': best, 'training_cv_macro_top1': values[best],
                  'training_cv_covered_fraction': float(covered.mean()),
                  'training_cv_scores': {str(k): v for k, v in values.items()}}


def evaluate_pair(partitions, labels, category):
    lab = labels[category]
    gids, _ = partitions['original_dinov3']['train']
    qids, _ = partitions['original_dinov3']['validation']
    gi = [i for i, name in enumerate(gids) if lab.get(name)]
    qi = [i for i, name in enumerate(qids) if lab.get(name)]
    gy = [set(lab[gids[i]]) for i in gi]
    qy = [set(lab[qids[i]]) for i in qi]
    vocabulary = sorted(set().union(*gy))
    folds = training_folds(gy)
    report = {'training_patients': len(gi), 'validation_patients': len(qi),
              'training_classes': len(vocabulary),
              'validation_classes': len(set().union(*qy)),
              'validation_label_coverage_fraction': float(np.mean([bool(y & set(vocabulary)) for y in qy])),
              'models': {}}
    hit_arrays = {}
    for name, parts in partitions.items():
        ids, x = parts['train']
        val_ids, q = parts['validation']
        if ids != gids or val_ids != qids:
            raise ValueError('patient_pair_mismatch')
        alpha, model = select_alpha(x[gi], gy, folds)
        scores = ridge_predictions(x[gi], q[qi], gy, vocabulary, [alpha])[alpha]
        hits = {k: label_hits(scores, qy, vocabulary, k) for k in (1, 5, 10)}
        model['validation'] = {f'top{k}_macro': macro_score(h, qy) for k, h in hits.items()}
        model['validation']['top1_micro'] = float(hits[1].mean())
        report['models'][name] = model
        hit_arrays[name] = hits[1]
    report['paired_top1'] = paired_interval(hit_arrays['original_dinov3'], hit_arrays['adapted'], qy,
                                             repetitions=2000, seed=SEED)
    return report


def main():
    os.umask(0o077)
    folder, out = BASE / 'validation', BASE / 'diagnostics'
    save(out / (REPORT_STEM + '_status.json'), {'state': 'running', 'started_at_utc': now(),
                                          'images_read': False, 'test_used': False})
    plan = load(folder / 'plan.json')
    for name, key in [('inputs.private.jsonl', 'inputs_sha256'),
                      ('labels.private.json', 'labels_sha256'),
                      ('original_baseline.private.npy', 'baseline_sha256')]:
        if digest(folder / name) != plan[key]:
            raise ValueError('frozen_input_integrity_changed')
    with (folder / 'inputs.private.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    if any(row['split'] not in ('train', 'validation') for row in records):
        raise ValueError('test_in_probe_inputs')
    ids = {split: {row['group_id'] for row in records if row['split'] == split}
           for split in ('train', 'validation')}
    if ids['train'] & ids['validation']:
        raise ValueError('patient_leakage')
    baseline = np.load(folder / 'original_baseline.private.npy', allow_pickle=False)
    current = np.empty_like(baseline)
    ranks = sorted((folder / 'step_003200').glob('rank_*.private.npy'))
    if len(ranks) != 12:
        raise ValueError('incomplete_final_features')
    for rank in range(12):
        value = np.load(folder / 'step_003200' / f'rank_{rank:02d}.private.npy', allow_pickle=False)
        if value.shape != current[rank::12].shape:
            raise ValueError('feature_shard_shape_mismatch')
        current[rank::12] = value
    partitions = {name: {split: aggregate_patients(records, values, split)
                         for split in ('train', 'validation')}
                  for name, values in [('original_dinov3', baseline), ('adapted', current)]}
    del baseline, current
    labels = load(folder / 'labels.private.json')['labels']
    report = {'state': 'running', 'protocol': 'patient_level_multilabel_balanced_ridge_cls',
              'hyperparameter_selection': 'five_fold_training_patients_only',
              'validation_used_for_selection': False, 'test_used': False, 'images_read': False,
              'label_ranking_not_patient_retrieval': True, 'seed': SEED,
              'alpha_candidates': list(ALPHAS),
              'grid_extension_reason': 'training_cv_optimum_on_upper_boundary_in_initial_grid',
              'metadata_labels_not_clinical_gold_standard': True, 'endpoints': {}}
    for category in ('disease_or_disorder', 'structural_diagnosis'):
        report['endpoints'][category] = evaluate_pair(partitions, labels, category)
        save(out / (REPORT_STEM + '_report.json'), report)
        save(out / (REPORT_STEM + '_status.json'), {'state': 'running', 'completed_endpoint': category,
                                               'updated_at_utc': now(), 'test_used': False})
    report.update(state='complete', completed_at_utc=now())
    save(out / (REPORT_STEM + '_report.json'), report)
    save(out / (REPORT_STEM + '_status.json'), {'state': 'complete', 'completed_at_utc': now(),
                                          'report_path': str(out / (REPORT_STEM + '_report.json')),
                                          'test_used': False, 'images_read': False})
    notify('frozen linear diagnostic complete', 'Frozen-encoder diagnostic completed',
           'The original and adapted encoders have been compared with training-only cross-validated frozen linear probes. This exploratory result does not establish clinical superiority. All detailed results remain on HPC.')
    print('Frozen-feature comparison completed; detailed results remain local.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(BASE / 'diagnostics' / (REPORT_STEM + '_status.json'), {'state': 'failed',
             'error_type': type(error).__name__, 'updated_at_utc': now()})
        raise
