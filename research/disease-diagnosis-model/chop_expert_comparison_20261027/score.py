"""Paired expert-patient retrieval with one fixed leave-patient-out gallery."""
import json,os,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def main():
    rows=load(ROOT/'private/released.json')['rows'];groups=collections.defaultdict(list);labels={}
    for i,r in enumerate(rows):
        groups[r['patient']].append(i)
        if r['patient'] in labels and labels[r['patient']]!=r['label']:raise RuntimeError('expert_patient_label_conflict')
        labels[r['patient']]=r['label']
    patients=sorted(groups);target=np.asarray([labels[p] for p in patients]);classes=np.unique(target)
    eligible=np.asarray([np.sum(target==t)>=2 for t in target]);ids=np.flatnonzero(eligible)
    if not len(ids):raise RuntimeError('no_multi_patient_expert_disease_for_retrieval')
    outcomes={};scores={}
    for arm in ['original','cpt','original_gmdb','cpt_gmdb']:
        if load(ROOT/(arm+'_receipt.json')).get('state')!='complete':raise RuntimeError('incomplete_model_features')
        f=np.load(ROOT/'private'/(arm+'.npy'));v=np.stack([f[groups[p]].mean(0) for p in patients]);v/=np.linalg.norm(v,axis=1,keepdims=True)
        similarities=v@v.T;np.fill_diagonal(similarities,-np.inf);hits=[]
        for i in ids:
            order=np.argsort(-similarities[i]);order=order[order!=i]
            ordered=target[order];ordered=ordered[np.sort(np.unique(ordered,return_index=True)[1])]
            hits.append([float(target[i] in ordered[:k]) for k in [1,5,10]])
        hits=np.asarray(hits);outcomes[arm]=hits
        scores[arm]={f'top{k}':float(np.mean([hits[target[ids]==c,j].mean() for c in np.unique(target[ids])])) for j,k in enumerate([1,5,10])}
    rng=np.random.default_rng(20261028);deltas={}
    for before,after in [('original','cpt'),('original_gmdb','cpt_gmdb')]:
        delta=outcomes[after]-outcomes[before];samples=np.zeros((2000,3));ds=np.unique(target[ids])
        for c in ds:
            v=delta[target[ids]==c];samples+=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)/len(ds)
        deltas[after+'_minus_'+before]={f'top{k}':dict(delta=scores[after][f'top{k}']-scores[before][f'top{k}'],paired_patient_bootstrap_95_interval=np.quantile(samples[:,j],[.025,.975]).tolist()) for j,k in enumerate([1,5,10])}
    result=dict(state='complete',setting='exploratory expert-source leave-one-patient-out gallery; patient-mean embeddings; single RGB view',
        query_patients=len(ids),gallery_patients=len(patients),query_diseases=len(np.unique(target[ids])),singleton_query_patients_excluded=int((~eligible).sum()),models=scores,paired_CPT_effects=deltas,
        expert_labels_used_for_training_or_model_selection=False,audit=load(ROOT/'release.json'),not_a_blinded_prospective_clinical_study=True)
    p=ROOT/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');os.chmod(p,0o600)
if __name__=='__main__':main()
