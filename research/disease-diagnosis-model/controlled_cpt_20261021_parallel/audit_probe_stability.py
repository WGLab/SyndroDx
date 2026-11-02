"""Post-hoc diagnostic of cached features; no images or model weights read.

Outputs contain aggregates only. Original trial reports and stop decisions are
immutable. The fixed-alpha sensitivity analysis is not a replacement endpoint.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='4'
import sys
import time
from pathlib import Path
WORK=Path(__file__).resolve().parent
BASE=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
DIAG=WORK.parent/'training_diagnostics_20261017'
sys.path[:0]=[str(WORK),str(DIAG/'vendor/dinov3_fino'),str(BASE),str(BASE.with_name('dinov3_craniofacial_v1')),str(DIAG)]
import numpy as np
from common import load,save,now,digest
from validation_hook import cohort
from evaluation_core import aggregate_patients,macro_score,paired_interval
from frozen_probe import ridge_predictions,label_hits

OUT=WORK/'probe_stability_audit'
ALPHAS=(1000.,10000.)


def quantiles(x):
    values=np.quantile(x,[0.,.01,.5,.99,1.])
    return dict(zip(('minimum','p01','median','p99','maximum'),map(float,values)))


def features(arm,step,shape):
    if arm=='original':
        return np.load(WORK/'original_runtime_baseline.private.npy',allow_pickle=False)
    folder=WORK/arm/'training/validation'/f'step_{step:06d}'
    result=np.empty(shape,dtype=np.float32)
    for rank in range(8):
        shard=np.load(folder/f'rank_{rank:02d}.private.npy',allow_pickle=False)
        if result[rank::8].shape!=shard.shape:raise RuntimeError('cached_feature_shape_mismatch')
        result[rank::8]=shard
    if not np.isfinite(result).all():raise RuntimeError('nonfinite_cached_features')
    return result


def main():
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    save(OUT/'status.json',dict(state='running',started_at_utc=now(),images_read=False,test_used=False))
    records,_,labels=cohort()
    baseline=np.load(WORK/'original_runtime_baseline.private.npy',allow_pickle=False)
    if len(records)!=len(baseline):raise RuntimeError('record_feature_count_mismatch')
    if digest(WORK/'original_runtime_baseline.private.npy')!=load(WORK/'original_runtime_baseline.json')['features_sha256']:
        raise RuntimeError('baseline_integrity_failure')
    cases=[('original',0),('random_heads',25),('random_heads',50),('warmed_heads',25)]
    reference_hits={};reference_parts=None;results=[]
    for arm,step in cases:
        started=time.monotonic()
        value=baseline if arm=='original' else features(arm,step,baseline.shape)
        parts={split:aggregate_patients(records,value,split) for split in ('train','validation')}
        if arm=='original':reference_parts=parts
        if any(parts[s][0]!=reference_parts[s][0] for s in parts):raise RuntimeError('patient_order_mismatch')
        if set(parts['train'][0])&set(parts['validation'][0]):raise RuntimeError('patient_split_overlap')
        baseline_norm=np.linalg.norm(baseline,axis=1)
        norms=np.linalg.norm(value,axis=1)
        cosine=np.sum(value*baseline,axis=1)/np.maximum(norms*baseline_norm,1e-12)
        patient_cosine={s:quantiles(np.sum(parts[s][1]*reference_parts[s][1],axis=1)) for s in parts}
        report=dict(arm=arm,step=step,image_cosine_to_original=quantiles(cosine),
                    patient_cosine_to_original=patient_cosine,feature_norms=quantiles(norms),
                    centered_feature_energy=float(np.mean(np.sum((value-value.mean(0))**2,axis=1))),
                    structural_ridge={})
        lab=labels['structural_diagnosis'];gids,g=parts['train'];qids,q=parts['validation']
        gi=[i for i,k in enumerate(gids) if lab.get(k)];qi=[i for i,k in enumerate(qids) if lab.get(k)]
        gy=[set(lab[gids[i]]) for i in gi];qy=[set(lab[qids[i]]) for i in qi]
        vocabulary=sorted(set().union(*gy))
        predictions=ridge_predictions(g[gi],q[qi],gy,vocabulary,ALPHAS)
        for alpha,prediction in predictions.items():
            hits=label_hits(prediction,qy,vocabulary,1)
            if arm=='original':reference_hits[alpha]=hits
            comparison=paired_interval(reference_hits[alpha],hits,qy,repetitions=2000)
            report['structural_ridge'][str(alpha)]=dict(top1_macro=macro_score(hits,qy),
                top1_micro=float(hits.mean()),changed_correctness_count=int(np.count_nonzero(hits!=reference_hits[alpha])),
                paired=comparison)
            if arm!='original':
                saved=load(WORK/arm/'training/validation'/f'step_{step:06d}/report.json')
                if saved['endpoints']['structural']['ridge']['selected_alpha']==alpha:
                    if abs(macro_score(hits,qy)-saved['metrics']['structural_ridge_top1'])>1e-12:
                        raise RuntimeError('saved_metric_not_reproduced')
        report['elapsed_seconds']=time.monotonic()-started
        results.append(report)
        save(OUT/'partial_aggregates.json',dict(state='running',results=results,images_read=False,test_used=False))
    original_at_1000=results[0]['structural_ridge']['1000.0']['top1_macro']
    adapted=next(x for x in results if x['arm']=='random_heads' and x['step']==50)
    fixed_delta=adapted['structural_ridge']['1000.0']['top1_macro']-original_at_1000
    tuned=load(WORK/'random_heads/training/validation/step_000050/report.json')['metrics']['structural_ridge_top1']
    save(OUT/'aggregate_report.json',dict(state='complete',results=results,
        random_step50_decomposition=dict(original_at_alpha1000=original_at_1000,
            adapted_at_alpha1000=adapted['structural_ridge']['1000.0']['top1_macro'],
            adapted_at_alpha10000=adapted['structural_ridge']['10000.0']['top1_macro'],
            original_at_alpha10000=results[0]['structural_ridge']['10000.0']['top1_macro'],
            fixed_alpha_representation_delta=fixed_delta,
            changed_alpha_delta=tuned-adapted['structural_ridge']['1000.0']['top1_macro']),
        post_hoc_sensitivity_analysis=True,original_reports_unchanged=True,
        no_hyperparameters_selected_from_validation=True,images_read=False,model_weights_read=False,
        test_used=False,patient_records_included=False,finished_at_utc=now()))
    save(OUT/'status.json',dict(state='complete',finished_at_utc=now(),images_read=False,test_used=False))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'status.json',dict(state='failed',error_type=type(e).__name__,updated_at_utc=now()))
        raise
