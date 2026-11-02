"""Local cached-feature audit; primary metrics and old decisions stay immutable."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import sys
import time
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK))
from configuration import BASE,DIAGNOSTICS
import numpy as np
from common import load,save,digest,now
from validation_hook import cohort
from evaluation_core import aggregate_patients,macro_score,retrieval,paired_interval
from frozen_probe import ridge_predictions,label_hits,training_folds

ALPHAS=(.01,.1,1.,10.,100.,1000.,10000.,100000.,1000000.)
OUT=WORK/'probe_audit'
OLD=WORK.with_name('cranio_only_cpt_20261021_single16')


def feature_case(step,shape):
    if step==0:return np.load(WORK/'original_runtime_baseline.private.npy',allow_pickle=False)
    source=WORK/'source_snapshot/validation/step_000575' if step==575 else OLD/'cranio_single16/training/validation'/f'step_{step:06d}'
    result=np.empty(shape,dtype=np.float32)
    for rank in range(16):
        shard=np.load(source/f'rank_{rank:02d}.private.npy',allow_pickle=False)
        if result[rank::16].shape!=shard.shape:raise RuntimeError('cached_feature_shape_mismatch')
        result[rank::16]=shard
    if not np.isfinite(result).all():raise RuntimeError('nonfinite_cached_features')
    return result


def training_cv(x,labels,vocabulary):
    fold=training_folds(labels)
    hits={a:np.zeros(len(labels),dtype=bool) for a in ALPHAS}
    for part in range(5):
        ti=np.flatnonzero(fold!=part);vi=np.flatnonzero(fold==part)
        ty=[labels[i] for i in ti];vy=[labels[i] for i in vi]
        words=sorted(set().union(*ty))
        predictions=ridge_predictions(x[ti],x[vi],ty,words,ALPHAS)
        for a,scores in predictions.items():hits[a][vi]=label_hits(scores,vy,words)
    values={str(a):macro_score(h,labels) for a,h in hits.items()}
    selected=max(ALPHAS,key=lambda a:(values[str(a)],a))
    return dict(selected_alpha=selected,training_cv_scores=values,validation_used_for_selection=False)


def main():
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    save(OUT/'status.json',dict(state='running',started_at_utc=now(),images_read=False,model_weights_read=False))
    records,_,labels=cohort()
    baseline=feature_case(0,None)
    if len(records)!=len(baseline) or digest(WORK/'original_runtime_baseline.private.npy')!=load(WORK/'original_runtime_baseline.json')['features_sha256']:
        raise RuntimeError('original_feature_integrity_failure')
    original_parts=None;references={};results=[];anchor_metrics={}
    for step in (0,500,525,550,575):
        started=time.monotonic();value=baseline if step==0 else feature_case(step,baseline.shape)
        parts={s:aggregate_patients(records,value,s) for s in ('train','validation')}
        if step==0:original_parts=parts
        if any(parts[s][0]!=original_parts[s][0] for s in parts):raise RuntimeError('patient_order_changed')
        gids,g=parts['train'];qids,q=parts['validation']
        if set(gids)&set(qids):raise RuntimeError('patient_split_overlap')
        cosine=np.sum(value*baseline,axis=1)/np.maximum(np.linalg.norm(value,axis=1)*np.linalg.norm(baseline,axis=1),1e-12)
        report=dict(step=step,minimum_cosine_to_original=float(cosine.min()),median_cosine_to_original=float(np.median(cosine)),
            feature_energy=float(np.mean(np.sum((value-value.mean(0))**2,axis=1))),endpoints={})
        for category,short,fixed in [('disease_or_disorder','disease',100.),('structural_diagnosis','structural',1000.)]:
            lab=labels[category]
            gi=[i for i,k in enumerate(gids) if lab.get(k)];qi=[i for i,k in enumerate(qids) if lab.get(k)]
            gy=[set(lab[gids[i]]) for i in gi];qy=[set(lab[qids[i]]) for i in qi]
            vocabulary=sorted(set().union(*gy))
            knn,coverage=retrieval(q[qi],g[gi],[qids[i] for i in qi],[gids[i] for i in gi],qy,gy)
            predictions=ridge_predictions(g[gi],q[qi],gy,vocabulary,ALPHAS)
            scores={};primary=None
            for alpha,prediction in predictions.items():
                hits=label_hits(prediction,qy,vocabulary)
                scores[str(alpha)]=macro_score(hits,qy)
                if alpha==fixed:primary=hits
            if step==0:references[short]=primary
            endpoint=dict(training_patients=len(gi),validation_patients=len(qi),
                fixed_alpha=fixed,fixed_ridge_top1=macro_score(primary,qy),retrieval_top1=macro_score(knn[1],qy),
                sensitivity_scores=scores,validation_results_not_used_to_pick_alpha=True,
                correctness_worsened_count=int(np.count_nonzero(references[short]&~primary)),
                correctness_improved_count=int(np.count_nonzero(~references[short]&primary)),
                feature_std_floor_fraction=float(np.mean(g[gi].std(0)<.01)))
            expected=load(WORK/'original_runtime_baseline.json')['metrics'] if step==0 else load((WORK/'source_snapshot/validation/step_000575/report.json') if step==575 else OLD/'cranio_single16/training/validation'/f'step_{step:06d}/report.json')['metrics']
            if abs(endpoint['fixed_ridge_top1']-expected[short+'_ridge_top1'])>1e-12 or abs(endpoint['retrieval_top1']-expected[short+'_retrieval_top1'])>1e-12:
                raise RuntimeError('primary_metric_reproduction_failed')
            if short=='disease' and step in (0,575):
                endpoint['training_only_cv']=training_cv(g[gi],gy,vocabulary)
                a=endpoint['training_only_cv']['selected_alpha']
                endpoint['training_cv_selected_alpha_validation_top1']=scores[str(a)]
            if step==575:
                np.savez(WORK/(short+'_resume_hits.private.npz'),retrieval=knn[1],ridge=primary)
                anchor_metrics[short+'_retrieval_top1']=endpoint['retrieval_top1']
                anchor_metrics[short+'_ridge_top1']=endpoint['fixed_ridge_top1']
            report['endpoints'][short]=endpoint
        report['elapsed_seconds']=time.monotonic()-started;results.append(report)
        save(OUT/'partial_aggregates.json',dict(state='running',results=results,patient_records_included=False))
    save(WORK/'resume_anchor.json',dict(state='passed',completed_steps=575,metrics=anchor_metrics,
        explicit_user_requested_resume=True,original_reports_unchanged=True,created_at_utc=now()))
    save(OUT/'aggregate_report.json',dict(state='complete',results=results,
        primary_metrics_reproduced=True,post_hoc_diagnostic_not_replacement_endpoint=True,
        images_read=False,model_weights_read=False,private_mapping_read=False,test_used=False,
        patient_records_included=False,finished_at_utc=now()))
    save(OUT/'status.json',dict(state='complete',primary_metrics_reproduced=True,finished_at_utc=now()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
