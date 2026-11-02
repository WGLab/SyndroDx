"""Aggregate development comparisons; no automatic promotion using test data."""
import numpy as np
from common import *

def main():
    p=load(WORK/'protocol.json');out={};effects={}
    for mode in p['modes']:
        rows=[load(WORK/'cases'/f'{mode}__seed{s}'/'result.json') for s in p['seeds']]
        assert all(r.get('state')=='complete' for r in rows)
        out[mode]={task:{f'top{k}_mean':float(np.mean([r['metrics'][task][f'top{k}'] for r in rows])) for k in [1,5,10]} for task in ['diagnosis','retrieval']}
        out[mode]['per_seed']={str(r['seed']):dict(metrics=r['metrics'],selected_epoch=r['selected_epoch']) for r in rows}
        out[mode]['controls']={control:{task:{f'top{k}_mean':float(np.mean([r['controls'][control][task][f'top{k}'] for r in rows])) for k in [1,5,10]} for task in rows[0]['controls'][control]} for control in rows[0]['controls']}
        files=[np.load(WORK/'cases'/f'{mode}__seed{s}'/'validation_hits.private.npz') for s in p['seeds']]
        labels=files[0]['labels'];assert all(np.array_equal(f['labels'],labels) for f in files)
        groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
        effects[mode]={}
        for baseline in ['hpo','visual','late_fusion']:
            effects[mode][baseline]={}
            for task in ['diagnosis','retrieval']:
                effects[mode][baseline][task]={}
                for k in [1,5]:
                    delta=np.stack([f[f'query_{task}_{k}'].astype(float)-f[f'{baseline}_{task}_{k}'].astype(float) for f in files])
                    rng=np.random.default_rng(20261101);boot=[]
                    for _ in range(2000):
                        seeds=rng.choice(len(delta),len(delta),replace=True)
                        classes=rng.choice(len(groups),len(groups),replace=True)
                        boot.append(np.mean([delta[seeds][:,rng.choice(groups[c],len(groups[c]),replace=True)].mean() for c in classes]))
                    effects[mode][baseline][task][f'top{k}']=dict(delta=float(np.mean([delta[:,g].mean() for g in groups])),ci95=np.quantile(boot,[.025,.975]).tolist())
        for f in files:f.close()
    save(WORK/'results.json',dict(state='complete',development_only=True,models=out,paired_effects=effects,
        interval_caveat='conditional on reused development-selected checkpoints; not selection-adjusted or confirmatory',
        locked_baseline_replaced=False,clinical_tests_used=False,updated_at_utc=now()))

if __name__=='__main__':main()
