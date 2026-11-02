"""CPU-only native initialization and frozen validation setup from local caches."""
import gc,sys,math,os
from pathlib import Path
from common import ROOT,V1,UPSTREAM,load,save,digest,now,notify,install_rope_dtype_compatibility
sys.path[:0]=[str(UPSTREAM),str(V1)]
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
from weights import OfficialBackbone


def configuration(batch=64,steps=2000,world=20):
 cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(ROOT/'official_vit7b_reference.yaml'))
 cfg.student.fp8_enabled=False;cfg.train.compile=False
 cfg.train.checkpointing=True;cfg.train.checkpointing_full=True
 cfg.train.batch_size_per_gpu=batch;cfg.train.num_workers=4;cfg.train.seed=20261017
 cfg.train.output_dir=str(ROOT/'training');cfg.train.dataset_path='released_local_manifests'
 cfg.train.OFFICIAL_EPOCH_LENGTH=1;cfg.optim.epochs=steps
 # Domain-specific LR only. Leave the upstream scheduler/scaling machinery intact.
 scale=4*math.sqrt(batch*world/1024)
 cfg.schedules.lr.start=0.;cfg.schedules.lr.peak=2e-6/scale;cfg.schedules.lr.end=2e-7/scale
 cfg.schedules.lr.warmup_epochs=max(10,steps//10)
 cfg.schedules.lr.freeze_last_layer_epochs=max(1,steps//200)
 cfg.schedules.teacher_temp.warmup_epochs=max(10,steps//10)
 cfg.student.resume_from_teacher_chkpt=str(ROOT/'initialization/teacher_dcp')
 cfg.checkpointing.period=100;cfg.checkpointing.max_to_keep=3;cfg.checkpointing.keep_every=10000
 cfg.evaluation.eval_period_iterations=100;cfg.train.sharded_eval_checkpoint=True
 return cfg


def validation_plan():
 folder=ROOT/'validation';folder.mkdir(exist_ok=True)
 src=V1/'evaluation';plan=load(src/'input_plan.json')
 if digest(src/'inputs.private.jsonl')!=plan['inputs_sha256']:raise RuntimeError('frozen_input_digest_changed')
 import json,collections
 with (src/'inputs.private.jsonl').open() as f:records=[json.loads(line) for line in f]
 labels=load(src/'labels.private.json');all_groups=set().union(*(set(v) for v in labels['labels'].values()))
 grouped=collections.defaultdict(list)
 for i,row in enumerate(records):
  if row['split'] in ['train','validation'] and row['group_id'] in all_groups:grouped[row['group_id']].append(i)
 selected=[]
 for group,indices in sorted(grouped.items()):
  chosen=[];views=set()
  for i in sorted(indices,key=lambda i:(records[i]['view'],records[i]['crop_sha256'])):
   if records[i]['view'] not in views:chosen.append(i);views.add(records[i]['view'])
   if len(chosen)==2:break
  if len(chosen)<2:
   chosen+= [i for i in indices if i not in chosen][:2-len(chosen)]
  selected.extend(chosen)
 if not selected:raise RuntimeError('no_validation_cohort')
 rows=[records[i] for i in selected]
 if not any(x['split']=='validation' for x in rows):raise RuntimeError('no_validation_queries')
 train={x['group_id'] for x in rows if x['split']=='train'};val={x['group_id'] for x in rows if x['split']=='validation'}
 if train&val:raise RuntimeError('validation_patient_overlap')
 # Verify source feature shards before reusing the unchanged official baseline.
 for slot in range(plan['world_size']):
  p=src/'features'/f'slot_{slot:02d}';receipt=load(p/'original_dinov3.json')
  if receipt.get('state')!='complete' or receipt.get('inputs_sha256')!=plan['inputs_sha256'] or digest(p/'original_dinov3.private.npy')!=receipt.get('features_sha256'):raise RuntimeError('baseline_shard_integrity_failed')
 # Read verified original shards without modifying the completed v1 run.
 position={index:j for j,index in enumerate(selected)};baseline=None;seen=np.zeros(len(selected),dtype=bool)
 for slot in range(plan['world_size']):
  shard=src/'features'/f'slot_{slot:02d}'
  indices=np.load(shard/'indices.private.npy',allow_pickle=False)
  values=np.load(shard/'original_dinov3.private.npy',mmap_mode='r',allow_pickle=False)
  if baseline is None:baseline=np.empty((len(selected),values.shape[1]),dtype=np.float32)
  for local,index in enumerate(indices):
   destination=position.get(int(index))
   if destination is not None:
    if seen[destination]:raise RuntimeError('baseline_partition_overlap')
    baseline[destination]=values[local];seen[destination]=True
 if not seen.all():raise RuntimeError('baseline_coverage_incomplete')
 np.save(folder/'original_baseline.private.npy',baseline)
 with (folder/'inputs.private.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 save(folder/'labels.private.json',labels)
 save(folder/'plan.json',dict(state='frozen_validation_only',inputs_sha256=digest(folder/'inputs.private.jsonl'),baseline_sha256=digest(folder/'original_baseline.private.npy'),labels_sha256=digest(folder/'labels.private.json'),source_inputs_sha256=plan['inputs_sha256'],test_used=False,cohort='train gallery and validation queries; up to two views per patient; fixed across checkpoints',created_at_utc=now()))
 del baseline;gc.collect()


def main():
 os.umask(0o077);torch.set_num_threads(4);torch.manual_seed(20261017)
 install_rope_dtype_compatibility()
 ready=ROOT/'initialization/ready.json';ready.parent.mkdir(exist_ok=True)
 cfg=configuration();OmegaConf.save(cfg,ROOT/'adaptation_template.yaml')
 validation_plan()
 save(ROOT/'preparation_status.json',dict(state='building_complete_native_teacher_initialization_on_CPU',updated_at_utc=now()))
 with torch.device('meta'):arch=SSLMetaArch(cfg)
 teacher=arch.teacher;del arch;gc.collect()
 teacher.to_empty(device='cpu');teacher.backbone.init_weights();teacher.dino_head.init_weights();teacher.ibot_head.init_weights()
 OfficialBackbone(V1/'assets/dinov3-vit7b16').load(teacher.backbone)
 target=ROOT/'initialization/teacher_dcp'
 if target.exists():raise RuntimeError('initialization_target_already_exists')
 dcp.save({'iteration':0,'model':teacher.state_dict()},storage_writer=dcp.FileSystemWriter(target))
 del teacher;gc.collect()
 save(ROOT/'preparation_status.json',dict(state='computing_public_synthetic_reference_on_CPU',updated_at_utc=now()))
 from transformers import AutoModel
 model=AutoModel.from_pretrained(str(V1/'assets/dinov3-vit7b16'),local_files_only=True,dtype=torch.float32,device_map={'':'cpu'},low_cpu_mem_usage=True).eval()
 image=torch.randn(1,3,224,224,generator=torch.Generator().manual_seed(8127))
 with torch.inference_mode():expected=model(pixel_values=image).last_hidden_state[:,0].float()
 torch.save({'input':image,'expected_cls':expected},ROOT/'initialization/public_synthetic_reference.pt')
 del model;gc.collect()
 save(ready,dict(state='ready',public_backbone='official cached ViT-7B',heads='freshly initialized at unchanged official reference dimensions; no pretrained SSL heads available',complete_native_teacher_dcp=True,not_an_original_full_SSL_checkpoint_resume=True,public_synthetic_reference_ready=True,validation_plan_frozen=True,completed_at_utc=now()))
 save(ROOT/'preparation_status.json',dict(state='ready',completed_at_utc=now()))
 notify('initialization prepared','Official-code CPT initialization ready','The local original-backbone initialization and fixed validation cohort are prepared. Official training code is retained. No clinical images, records, embeddings, private weights or metrics are included in this email.')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  traceback.print_exc()
  save(ROOT/'preparation_status.json',dict(state='failed',error_type=type(exc).__name__,updated_at_utc=now()))
  notify('preparation failure','Official-code CPT preparation requires attention','Local preparation failed before training started. Existing models and checkpoints are preserved; all detailed diagnostics remain on HPC.')
  raise SystemExit(1)
