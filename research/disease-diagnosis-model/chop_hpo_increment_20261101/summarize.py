"""Matched four-seed development effects using one stable ranking convention."""
import numpy as np
from common import *


def main():
    protocol = load(WORK / 'protocol.json')
    with np.load(WORK / 'private/gmdb.npz') as d:
        labels = d['labels'][d['validation']]
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    def macro(a): return float(np.mean([a[...,g].mean() for g in groups]))
    summaries, hits = {}, {}
    for condition in protocol['conditions']:
        reports = [load(WORK / 'cases' / f'{condition}__seed{s}' / 'result.json') for s in protocol['seeds']]
        assert all(r['state'] == 'complete' for r in reports)
        ranks = [np.argsort(-np.load(WORK / 'cases' / f'{condition}__seed{s}' / 'validation_logits.private.npy'), axis=1, kind='stable') for s in protocol['seeds']]
        summaries[condition] = {}
        hits[condition] = {}
        for k in [1,5,10]:
            h = np.stack([(r[:,:k] == labels[:,None]).any(1) for r in ranks])
            hits[condition][k] = h
            values = [macro(x) for x in h]
            assert np.allclose(values, [r['validation'][f'macro_top{k}'] for r in reports], atol=1e-6)
            summaries[condition][f'top{k}_mean'] = float(np.mean(values))
            summaries[condition][f'top{k}_SD'] = float(np.std(values, ddof=1))
    effects = {}
    for control in ['no_chop','chop_shuffled']:
        effects[control] = {}
        for k in [1,5]:
            delta = hits['chop_recorded'][k].astype(float) - hits[control][k].astype(float)
            rng = np.random.default_rng(20261101)
            values = []
            for _ in range(2000):
                seed = rng.choice(len(delta), len(delta), replace=True)
                selected = rng.choice(len(groups), len(groups), replace=True)
                values.append(np.mean([delta[seed][:,rng.choice(groups[c],len(groups[c]),replace=True)].mean() for c in selected]))
            effects[control][f'top{k}'] = dict(delta=macro(delta),ci95=np.quantile(values,[.025,.975]).tolist())
    save(WORK / 'results.json',dict(state='complete',summary=summaries,CHOP_increment_paired=effects,
        scope='GMDB patient-disjoint development only; same four seeds, initialization, update budget and stable ranking',
        CHOP_supervision='weak clinician-recorded patient association; unverified polarity, no CHOP diagnosis labels',
        clinical_test_used=False,visual_backbone_frozen=True,updated_at_utc=now()))


if __name__ == '__main__': main()
