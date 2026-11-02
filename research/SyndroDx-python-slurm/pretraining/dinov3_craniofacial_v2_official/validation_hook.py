"""Validation-only inference added after upstream teacher checkpoint saving.
No training targets, gradients, EMA or optimizer updates are changed.
"""
import sys,json,math
from pathlib import Path
from common import ROOT,V1,load,save,digest,now,notify
sys.path.insert(0,str(V1))
import numpy as np
import torch
import torch.distributed as dist
from evaluation_core import aggregate_patients,retrieval,macro_score,paired_interval
from evaluation_worker import fit_image
_CACHE=None

def cohort():
 global _CACHE
 if _CACHE is None:
  folder=ROOT/'validation';plan=load(folder/'plan.json')
  for name,key in [('inputs.private.jsonl','inputs_sha256'),('original_baseline.private.npy','baseline_sha256'),('labels.private.json','labels_sha256')]:
   if digest(folder/name)!=plan[key]:raise RuntimeError('frozen_validation_input_changed')
  with (folder/'inputs.private.jsonl').open() as f:records=[json.loads(x) for x in f]
  if any(x['split'] not in ['train','validation'] for x in records):raise RuntimeError('test_in_validation_cohort')
  baseline=np.load(folder/'original_baseline.private.npy',mmap_mode='r',allow_pickle=False)
  _CACHE=records,baseline,load(folder/'labels.private.json')['labels']
 return _CACHE


def compare(records,baseline,current,labels,step):
 partitions={kind:{split:aggregate_patients(records,features,split) for split in ['train','validation']} for kind,features in [('original_dinov3',baseline),('adapted',current)]}
 report=dict(state='complete',completed_steps=step,split='validation_only',test_used=False,metadata_labels_not_clinical_gold_standard=True,endpoints={},completed_at_utc=now())
 for category in ['disease_or_disorder','structural_diagnosis']:
  lab=labels.get(category,{});endpoint=dict(models={});results={}
  for kind,parts in partitions.items():
   gid,g=parts['train'];qid,q=parts['validation'];gi=[i for i,k in enumerate(gid) if lab.get(k)];qi=[i for i,k in enumerate(qid) if lab.get(k)]
   if not gi or not qi:endpoint['models'][kind]=dict(state='not_evaluable');continue
   gy=[set(lab[gid[i]]) for i in gi];qy=[set(lab[qid[i]]) for i in qi]
   hits,covered=retrieval(q[qi],g[gi],[qid[i] for i in qi],[gid[i] for i in gi],qy,gy)
   endpoint['models'][kind]=dict(state='complete',top1_macro=macro_score(hits[1],qy),top5_macro=macro_score(hits[5],qy),top10_macro=macro_score(hits[10],qy),gallery_coverage=float(covered.mean()))
   results[kind]=(hits[1],qy)
  if len(results)==2:
   first,terms=results['original_dinov3'];second,_=results['adapted']
   endpoint['paired_top1']=paired_interval(first,second,terms,repetitions=1000)
  report['endpoints'][category]=endpoint
 return report


def validate(cfg,model,iteration):
 rank=dist.get_rank();world=dist.get_world_size();step=int(str(iteration).split('_')[-1])+1
 folder=ROOT/'validation'/f'step_{step:06d}';folder.mkdir(parents=True,exist_ok=True)
 if rank==0:save(ROOT/'phase.json',dict(state='validation_inference',completed_steps=step,world_size=world,updated_at_utc=now()))
 records,baseline,labels=cohort();indices=list(range(rank,len(records),world));batch=16
 outputs=np.empty((len(indices),model.embed_dim),dtype=np.float32)
 maxlocal=math.ceil(len(records)/world)
 for start in range(0,maxlocal,batch):
  selected=indices[start:start+batch];valid=torch.ones((),device='cuda',dtype=torch.int)
  try:
   tensors=[]
   for i in selected:
    if digest(records[i]['crop_path'])!=records[i]['crop_sha256']:raise RuntimeError('released_validation_crop_changed')
    tensors.append(fit_image(records[i]['crop_path'],cfg))
   while len(tensors)<batch:tensors.append(torch.zeros(3,cfg.crops.global_crops_size,cfg.crops.global_crops_size))
   inputs=torch.stack(tensors).cuda(non_blocking=True)
  except Exception:
   valid.zero_();inputs=torch.zeros(batch,3,cfg.crops.global_crops_size,cfg.crops.global_crops_size,device='cuda')
  dist.all_reduce(valid,op=dist.ReduceOp.MIN)
  if not bool(valid):raise RuntimeError('local_validation_input_unavailable')
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   value=model.teacher.backbone(inputs,is_training=True)['x_norm_clstoken'].float()
   value=torch.nn.functional.normalize(value,dim=-1)
  finite=torch.isfinite(value).all().to(torch.int);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
  if not bool(finite):raise RuntimeError('validation_nonfinite_features')
  if selected:outputs[start:start+len(selected)]=value[:len(selected)].cpu().numpy()
  del inputs,value
 np.save(folder/f'rank_{rank:02d}.private.npy',outputs)
 dist.barrier()
 if rank==0:
  current=np.empty((len(records),model.embed_dim),dtype=np.float32)
  for i in range(world):current[i::world]=np.load(folder/f'rank_{i:02d}.private.npy',allow_pickle=False)
  report=compare(records,baseline,current,labels,step);save(folder/'report.private.json',report)
  pair=report['endpoints'].get('disease_or_disorder',{}).get('paired_top1',{});ci=pair.get('paired_patient_bootstrap_95_interval')
  outcome='improvement_supported' if ci and ci[0]>0 else 'decline_supported' if ci and ci[1]<0 else 'improvement_not_established'
  receipt=dict(state='complete',completed_steps=step,primary_qualitative_outcome=outcome,report_path=str(folder/'report.private.json'),test_used=False,completed_at_utc=now())
  save(ROOT/'validation/latest.json',receipt)
  if step==100:notify('first validation complete','Official-code CPT early validation completed','The first scheduled validation comparison with the unchanged original encoder is complete. Training continues with the declared configuration. Detailed results remain on HPC.')
  if outcome=='decline_supported':notify('validation decline '+str(step),'Official-code CPT validation requires review','The validation-only comparison indicates a decline on the primary metadata endpoint. The saved checkpoint and original control are preserved. The trainer has not silently changed its recipe; detailed results remain on HPC.')
  save(ROOT/'phase.json',dict(state='training',completed_steps=step,world_size=world,updated_at_utc=now()))
 dist.barrier()


def install(official):
 original=official.do_test
 def checkpoint_and_validate(cfg,model,iteration,process_group,do_low_freq=False):
  original(cfg,model,iteration,process_group,do_low_freq=do_low_freq)
  validate(cfg,model,iteration)
 official.do_test=checkpoint_and_validate
