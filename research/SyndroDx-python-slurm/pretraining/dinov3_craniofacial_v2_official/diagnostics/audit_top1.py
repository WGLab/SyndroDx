"""Aggregate-only diagnosis of existing frozen validation embeddings; no images."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import sys,json,collections
from pathlib import Path
import numpy as np
BASE=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1=BASE.parent/'dinov3_craniofacial_v1';sys.path.insert(0,str(V1));sys.path.insert(0,str(BASE))
from common import load,save,now,digest
from evaluation_core import aggregate_patients,normalized,collapse_statistics,macro_score

def main():
 folder=BASE/'validation';plan=load(folder/'plan.json')
 for n,k in [('inputs.private.jsonl','inputs_sha256'),('labels.private.json','labels_sha256'),('original_baseline.private.npy','baseline_sha256')]:
  if digest(folder/n)!=plan[k]:raise RuntimeError('frozen_input_integrity_changed')
 with (folder/'inputs.private.jsonl').open() as f:records=[json.loads(line) for line in f]
 assert all(a['split'] in ['train','validation'] for a in records)
 before=np.load(folder/'original_baseline.private.npy',allow_pickle=False)
 after=np.empty_like(before);final=folder/'step_003200';world=12
 for rank in range(world):
  x=np.load(final/f'rank_{rank:02d}.private.npy',allow_pickle=False);assert x.shape==after[rank::world].shape;after[rank::world]=x
 labels=load(folder/'labels.private.json')['labels']
 p={name:{split:aggregate_patients(records,values,split) for split in ['train','validation']} for name,values in [('baseline',before),('adapted',after)]}
 report=dict(state='complete',images_read=False,only_existing_embeddings_used=True,test_used=False,completed_at_utc=now(),drift={},retrieval={})
 for split in ['train','validation']:
  ids,a=p['baseline'][split];other,b=p['adapted'][split];assert ids==other
  cosine=(a*b).sum(1)
  report['drift'][split]=dict(patients=len(ids),paired_cosine_mean=float(cosine.mean()),paired_cosine_p05=float(np.quantile(cosine,.05)),baseline_collapse_statistics=collapse_statistics(a),adapted_collapse_statistics=collapse_statistics(b))
 for category in ['disease_or_disorder','structural_diagnosis']:
  lab=labels.get(category,{})
  gids,_=p['baseline']['train'];qids,_=p['baseline']['validation'];gi=[i for i,k in enumerate(gids) if lab.get(k)];qi=[i for i,k in enumerate(qids) if lab.get(k)]
  gy=[set(lab[gids[i]]) for i in gi];qy=[set(lab[qids[i]]) for i in qi];ranks={}
  for kind in ['baseline','adapted']:
   _,g=p[kind]['train'];_,q=p[kind]['validation'];scores=q[qi]@g[gi].T
   found=[];margins=[]
   for row,y in zip(scores,qy):
    order=np.argsort(-row,kind='stable');matching=np.array([bool(y&gy[int(i)]) for i in order]);where=np.flatnonzero(matching)
    found.append(int(where[0])+1 if len(where) else len(gi)+1);margins.append(float(row[order[0]]-row[order[1]]))
   ranks[kind]=np.asarray(found);report['retrieval'].setdefault(category,{})[kind]=dict(queries=len(qi),gallery_patients=len(gi),top1_correct=int((ranks[kind]==1).sum()),top5_correct=int((ranks[kind]<=5).sum()),median_first_correct_rank=float(np.median(ranks[kind])),mean_top1_top2_cosine_gap=float(np.mean(margins)))
  b,a=ranks['baseline'],ranks['adapted'];lost=(b==1)&(a!=1);gained=(b!=1)&(a==1)
  counts=collections.Counter(term for terms in qy for term in terms)
  report['retrieval'][category]['paired_transitions']=dict(top1_both_correct=int(((b==1)&(a==1)).sum()),top1_lost=int(lost.sum()),top1_gained=int(gained.sum()),lost_top1_still_in_top5=int((lost&(a<=5)).sum()),validation_label_classes=len(counts),classes_with_one_validation_patient=sum(v==1 for v in counts.values()))
 save(BASE/'diagnostics/top1_audit_20261017.json',report)
 print(json.dumps(report))

if __name__=='__main__':main()
