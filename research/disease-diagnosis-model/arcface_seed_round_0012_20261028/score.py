"""Aggregate-only scoring of frozen encoders. Never uses expert or test records."""
import os,sys,collections
import numpy as np
from matched_runtime import *
from data import evaluation_cohort
sys.path.insert(0,str(V1));sys.path.insert(0,str(PROJECT/'training_diagnostics_20261017'))
from evaluation_core import normalized,aggregate_patients,retrieval,macro_score,collapse_statistics,paired_interval
from frozen_probe import select_alpha,training_folds,ridge_predictions,label_hits

def fewshot(x,q,y,qy):
    result={}
    for k in [1,5,10]:
        grouped={d:np.array([i for i,v in enumerate(y) if v==d]) for d in sorted(set(y))}
        eligible=[d for d,ids in grouped.items() if len(ids)>=k]
        qi=[i for i,d in enumerate(qy) if d in eligible]
        if len(eligible)<2 or not qi:
            result[str(k)]=dict(state='not_evaluable',eligible_diseases=len(eligible),query_patients=len(qi));continue
        rng=np.random.default_rng(20261028+k);values={n:[] for n in [1,5,10]}
        for repeat in range(20):
            prototypes=normalized(np.stack([x[rng.choice(grouped[d],size=k,replace=False)].mean(0) for d in eligible]))
            ranks=np.argsort(-(q[qi]@prototypes.T),axis=1,kind='stable')
            for n in values:
                hit=np.array([qy[i] in {eligible[j] for j in ranks[z,:n]} for z,i in enumerate(qi)])
                values[n].append(macro_score(hit,[{qy[i]} for i in qi]))
        result[str(k)]=dict(state='complete',support_patients_per_disease=k,eligible_diseases=len(eligible),query_patients=len(qi),
            repeats=20,**{f'macro_top{n}':float(np.mean(v)) for n,v in values.items()},
            uncertainty='repeated fixed support draws; not a clinical confidence interval',patient_disjoint=True)
    return result

def hpo_score(x,q,tr,va):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score,roc_auc_score
    ontology=load(PAIRS/'private/ontology.json');terms=ontology['terms'];facial=set(ontology['facial_roots'])
    while True:
        extra={k for k,v in terms.items() if set(v['parents'])&facial}-facial
        if not extra:break
        facial.update(extra)
    vocabulary=sorted(set().union(*(set(p['hpo']) for p in tr))&facial)
    aps=[];aucs=[];counts=[];prevalence=[]
    mean=x.mean(0);std=np.maximum(x.std(0),.01);x=(x-mean)/std;q=(q-mean)/std
    for term in vocabulary:
        ti=[i for i,p in enumerate(tr) if term in p['all_present_hpo'] or term in p['absent_hpo']]
        vi=[i for i,p in enumerate(va) if term in p['all_present_hpo'] or term in p['absent_hpo']]
        ty=np.array([term in tr[i]['all_present_hpo'] for i in ti],dtype=int)
        vy=np.array([term in va[i]['all_present_hpo'] for i in vi],dtype=int)
        if ty.sum()<10 or len(ty)-ty.sum()<5 or vy.sum()<2 or len(vy)-vy.sum()<2:continue
        clf=LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=20261028).fit(x[ti],ty)
        scores=clf.decision_function(q[vi]);aps.append(average_precision_score(vy,scores));aucs.append(roc_auc_score(vy,scores));prevalence.append(float(vy.mean()))
        counts.append(dict(training_known=len(ti),validation_known=len(vi),validation_positive=int(vy.sum())))
    return dict(state='complete' if aps else 'not_evaluable',eligible_terms=len(aps),
        macro_auprc=float(np.mean(aps)) if aps else None,macro_auroc=float(np.mean(aucs)) if aucs else None,
        prevalence_only_macro_auprc=float(np.mean(prevalence)) if prevalence else None,
        macro_auprc_gain_over_prevalence=float(np.mean(np.array(aps)-np.array(prevalence))) if aps else None,
        documented_assertions_only=True,missing_labels_used_as_negative=False,
        minimum_support='train >=10 positive and >=5 explicit negative; validation >=2 of each',
        known_assertion_pairs=sum(c['validation_known'] for c in counts),
        limitation='annotation availability limits coverage; this does not localize morphology')

def internal_score(rows,f,labels):
    parts={s:aggregate_patients(rows,f,s) for s in ['train','validation']};gids,g=parts['train'];qids,q=parts['validation'];out={}
    for key in ['disease_or_disorder','structural_diagnosis']:
        lab=labels[key];gi=[i for i,k in enumerate(gids) if lab.get(k)];qi=[i for i,k in enumerate(qids) if lab.get(k)]
        gy=[set(lab[gids[i]]) for i in gi];qy=[set(lab[qids[i]]) for i in qi]
        hits,covered=retrieval(q[qi],g[gi],[qids[i] for i in qi],[gids[i] for i in gi],qy,gy)
        alpha,_=select_alpha(g[gi],gy,training_folds(gy));vocab=sorted(set().union(*gy))
        pred=ridge_predictions(g[gi],q[qi],gy,vocab,[alpha])[alpha]
        out[key]=dict(training_patients=len(gi),query_patients=len(qi),
            retrieval={f'top{k}':macro_score(hits[k],qy) for k in [1,5,10]},
            ridge={f'top{k}':macro_score(label_hits(pred,qy,vocab,k),qy) for k in [1,5,10]},
            ridge_alpha=alpha,alpha_selection='training-only 5-fold CV')
    return out,collapse_statistics(q)

def main():
    os.umask(0o077);folder=Path(sys.argv[1]);ready=load(folder/'ready.json')
    assert ready and ready['protocol_sha256']==digest(ROOT/'protocol.json')
    p=load(ROOT/'protocol.json');assert digest(ROOT/'private/gmdb_patients.json')==p['gmdb_patient_sha256']
    rows,labels=evaluation_cohort();patients=load(ROOT/'private/gmdb_patients.json')['patients'];indices=sorted({i for p in patients for i in p['image_indices']})
    arrays={}
    for name,n in [('internal',len(rows)),('gmdb',len(indices))]:
        f=np.empty((n,512),np.float32)
        for r in range(ready['world']):
            shard=np.load(folder/f'{name}_{r}.npy',allow_pickle=False)
            assert shard.shape==f[r::ready['world']].shape;f[r::ready['world']]=shard
        arrays[name]=normalized(f)
    internal,health=internal_score(rows,arrays['internal'],labels)
    lookup={i:j for j,i in enumerate(indices)}
    feature=normalized(np.stack([arrays['gmdb'][[lookup[i] for i in pt['image_indices']]].mean(0) for pt in patients]))
    ti=[i for i,pt in enumerate(patients) if pt['split']=='train'];vi=[i for i,pt in enumerate(patients) if pt['split']=='validation']
    tr=[patients[i] for i in ti];va=[patients[i] for i in vi];x,q=feature[ti],feature[vi];y=[pt['label'] for pt in tr];qy=[pt['label'] for pt in va]
    hits,coverage=retrieval(q,x,[pt['subject'] for pt in va],[pt['subject'] for pt in tr],[{v} for v in qy],[{v} for v in y])
    np.savez(folder/'private_hits.npz',**{f'top{k}':hits[k] for k in hits})
    report=dict(state='complete',step=ready['step'],arm=folder.parent.parent.name,
        gmdb_development=dict(training_patients=len(ti),query_patients=len(vi),query_diseases=len(set(qy)),
            gallery_covered_patients=int(coverage.sum()),retrieval={f'top{k}':macro_score(hits[k],[{v} for v in qy]) for k in hits},
            fewshot=fewshot(x,q,y,qy),hpo_prediction=hpo_score(x,q,tr,va)),
        internal_development=internal,feature_health=health,
        test_used=False,expert_used=False,strict_unseen_image_claim=False,updated_at_utc=now())
    base=folder.parent/'step_0000000'
    if ready['step'] and (base/'results.json').exists():
        orig=np.load(base/'private_hits.npz',allow_pickle=False)
        report['gmdb_paired_to_original']={f'top{k}':paired_interval(orig[f'top{k}'],hits[k],[{v} for v in qy],repetitions=1000) for k in [1,5,10]}
        basehealth=load(base/'results.json')['feature_health']
        report['severe_collapse']=health['total_coordinate_variance']<.25*basehealth['total_coordinate_variance'] or health['effective_rank']<5
    save(folder/'results.json',report)
    print('Development scoring complete; only aggregate report exported locally.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(Path(sys.argv[1])/'score_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
