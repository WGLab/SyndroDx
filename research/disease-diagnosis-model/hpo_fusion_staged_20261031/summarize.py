"""Paired development diagnosis comparisons. No clinical test selection."""
import collections
import numpy as np
from common import *


def interval(a, b, labels, repeats=2000):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    def macro(x):
        return float(np.mean([x[..., g].mean() for g in groups]))
    rng = np.random.default_rng(20261031)
    values = []
    for _ in range(repeats):
        seeds = rng.choice(a.shape[0], a.shape[0], replace=True)
        chosen = rng.choice(len(groups), len(groups), replace=True)
        sample = []
        for c in chosen:
            patients = rng.choice(groups[c], len(groups[c]), replace=True)
            sample.append((b[seeds][:, patients] - a[seeds][:, patients]).mean())
        values.append(np.mean(sample))
    return dict(delta=macro(b) - macro(a), ci95=np.quantile(values, [.025, .975]).tolist(),
                resampling='paired seed/disease/patient bootstrap, development only', repetitions=repeats)


def main():
    protocol = load(WORK / 'protocol.json')
    cases = load(WORK / 'cases.json')['cases']
    values = {n: load(WORK / 'cases' / n / 'result.json') for n in cases}
    assert all(v.get('state') == 'complete' for v in values.values())
    with np.load(WORK / 'private/gmdb.npz') as d:
        labels = d['labels'][d['validation']]
    summary, paired, all_hits = {}, {}, {}
    for condition in protocol['conditions']:
        summary[condition] = {}
        hits = {}
        for mode in protocol['modes']:
            names = [f'{condition}__{mode}__seed{s}' for s in protocol['seeds']]
            summary[condition][mode] = {f'macro_top{k}_{label}': float(fn([values[n]['validation'][f'macro_top{k}'] for n in names]))
                for k in (1, 5, 10) for label, fn in [('mean', np.mean), ('SD', lambda a: np.std(a, ddof=1))]}
            hits[mode] = {}
            for k in (1, 5):
                scores = [np.load(WORK / 'cases' / n / 'validation_logits.private.npy') for n in names]
                hits[mode][k] = np.stack([(np.argsort(-s, axis=1)[:, :k] == labels[:, None]).any(1) for s in scores])
            if mode != 'hpo':
                summary[condition][mode]['shuffled_face_macro_top1_mean'] = float(np.mean([values[n]['controls']['shuffled_face']['macro_top1'] for n in names]))
        all_hits[condition] = hits
        paired[condition] = {mode: {f'top{k}': interval(hits['hpo'][k], hits[mode][k], labels) for k in (1, 5)} for mode in protocol['modes'] if mode != 'hpo'}
    resource_effect = {c: {m: {f'top{k}': interval(all_hits['gmdb_only'][m][k], all_hits[c][m][k], labels) for k in (1, 5)} for m in protocol['modes']} for c in protocol['conditions'] if c != 'gmdb_only'}
    save(WORK / 'results.json', dict(state='complete', summary=summary, facial_increment_paired=paired, public_data_increment_paired=resource_effect,
        scope='GMDB patient-disjoint development, new cohort; not GMDB test/CHOP/BCH',
        all_four_seeds_retained=True, visual_backbone_frozen=True, data_receipt=load(WORK / 'data_receipt.json'),
        clinical_test_used=False, updated_at_utc=now()))


if __name__ == '__main__':
    main()
