"""Cached-feature geometry checks invariant to global coordinate rotations."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from scipy.stats import spearmanr
from audit_probe import WORK,OUT,feature_case,cohort,aggregate_patients,save,now
from common import notify


def main():
    records,_,_=cohort()
    baseline=feature_case(0,None);current=feature_case(575,baseline.shape)
    before={s:aggregate_patients(records,baseline,s) for s in ('train','validation')}
    after={s:aggregate_patients(records,current,s) for s in ('train','validation')}
    if any(before[s][0]!=after[s][0] for s in before):raise RuntimeError('patient_alignment_changed')
    x=before['train'][1].astype(np.float64);y=after['train'][1].astype(np.float64)
    q=before['validation'][1].astype(np.float64);r=after['validation'][1].astype(np.float64)
    rng=np.random.default_rng(20261022)
    selected=rng.choice(len(x),min(512,len(x)),replace=False)
    a=x[selected];b=y[selected]
    ac=a-a.mean(0);bc=b-b.mean(0)
    ka=ac@ac.T;kb=bc@bc.T
    cka=float(np.sum(ka*kb)/np.sqrt(np.sum(ka*ka)*np.sum(kb*kb)))
    upper=np.triu_indices(len(a),1)
    correlation=float(spearmanr((a@a.T)[upper],(b@b.T)[upper]).statistic)
    if not np.isfinite(cka) or not np.isfinite(correlation):raise RuntimeError('invalid_geometry_aggregate')
    centered={}
    for name,left,right in [('train',x,y),('validation',q,r)]:
        left=left-x.mean(0);right=right-y.mean(0)
        cosine=(left*right).sum(1)/np.maximum(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1),1e-12)
        centered[name]=float(np.median(cosine))
    old_neighbor=np.argmax(q@x.T,axis=1);new_neighbor=np.argmax(r@y.T,axis=1)
    result=dict(state='complete',source_completed_steps=575,patient_sample_for_geometry=len(a),
        linear_centered_kernel_alignment=cka,pairwise_cosine_spearman=correlation,
        paired_feature_cosine_after_training_mean_removal=centered,
        training_mean_norm_original=float(np.linalg.norm(x.mean(0))),
        training_mean_norm_adapted=float(np.linalg.norm(y.mean(0))),
        nearest_patient_identity_agreement=float(np.mean(old_neighbor==new_neighbor)),
        labels_used_for_geometry=False,images_read=False,test_used=False,
        clinical_utility_not_established_by_geometry=True,finished_at_utc=now())
    save(OUT/'geometry_aggregate_report.json',result)
    notify('resume20 geometry audit complete','Craniofacial feature geometry diagnostic completed',
        'The cached-feature geometry diagnostic completed locally. Aggregate comparisons are saved on HPC and do not establish clinical superiority. No images, patient records or identifiers are included in this notification.')


if __name__=='__main__':main()
