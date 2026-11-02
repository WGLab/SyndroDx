"""Synthetic contract checks; no clinical files, images or weights are read."""
import tempfile,unittest,json,sys
from pathlib import Path
from unittest.mock import patch
from common import ROOT,UPSTREAM,V1,digest
sys.path[:0]=[str(UPSTREAM),str(V1)]
import numpy as np
import torch
from omegaconf import OmegaConf
from prepare import configuration
from validation_hook import compare
import validation_hook

class Contracts(unittest.TestCase):
 def test_full_official_head_dimensions_are_preserved(self):
  cfg=configuration(64,2000)
  self.assertEqual(cfg.dino.head_n_prototypes,262144);self.assertEqual(cfg.dino.head_hidden_dim,8192)
  self.assertEqual(cfg.ibot.head_n_prototypes,98304);self.assertEqual(cfg.ibot.head_bottleneck_dim,384)
  self.assertEqual(cfg.evaluation.eval_period_iterations,100);self.assertEqual(cfg.checkpointing.period,100)
 def test_lr_uses_official_scheduler_with_declared_effective_rate(self):
  import dinov3.train.train as official
  with patch.object(official.distributed,'get_world_size',return_value=20):schedules=official.build_schedulers(configuration(64,2000))
  self.assertAlmostEqual(max(schedules[0]),2e-6,places=10);self.assertEqual(len(schedules[0]),2000)
 def test_world_resize_preserves_iteration_schedule_and_effective_lr(self):
  import dinov3.train.train as official
  from configure_run import resumed_configuration
  cfg=configuration(64,3200,world=12)
  before=OmegaConf.to_container(cfg,resolve=True)
  resumed=resumed_configuration(cfg,64,20)
  self.assertEqual(OmegaConf.to_container(cfg,resolve=True),before)
  self.assertEqual(resumed.optim.epochs,3200)
  self.assertEqual(resumed.schedules.lr.warmup_epochs,cfg.schedules.lr.warmup_epochs)
  self.assertEqual(resumed.schedules.teacher_temp,cfg.schedules.teacher_temp)
  self.assertEqual(resumed.schedules.momentum,cfg.schedules.momentum)
  with patch.object(official.distributed,'get_world_size',return_value=20):schedules=official.build_schedulers(resumed)
  self.assertAlmostEqual(max(schedules[0]),2e-6,places=10)
  self.assertEqual(len(schedules[0]),3200)
 def test_validation_detects_known_improvement_on_same_cases(self):
  rows=[];baseline=[];updated=[];labels={}
  for split in ['train','validation']:
   for i in range(16):
    label='A' if i%2==0 else 'B';group=split+str(i);labels[group]=[label]
    vector=np.array([1.,0.] if label=='A' else [0.,1.],np.float32)
    rows.append(dict(group_id=group,split=split,view='frontal'))
    updated.append(vector);baseline.append(vector if split=='train' else vector[::-1])
  report=compare(rows,np.array(baseline),np.array(updated),{'disease_or_disorder':labels},100)
  self.assertFalse(report['test_used']);pair=report['endpoints']['disease_or_disorder']['paired_top1']
  self.assertEqual(pair['delta_macro'],1.);self.assertEqual(pair['paired_patient_bootstrap_95_interval'],[1.,1.])
 def test_validation_cohort_rejects_test_records(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'validation';folder.mkdir()
   (folder/'inputs.private.jsonl').write_text(json.dumps(dict(split='test',group_id='synthetic'))+'\n')
   (folder/'labels.private.json').write_text('{}');np.save(folder/'original_baseline.private.npy',np.zeros((1,2),np.float32))
   (folder/'plan.json').write_text(json.dumps({key:digest(folder/name) for name,key in [('inputs.private.jsonl','inputs_sha256'),('labels.private.json','labels_sha256'),('original_baseline.private.npy','baseline_sha256')]}))
   with patch.object(validation_hook,'ROOT',root),patch.object(validation_hook,'_CACHE',None):
    with self.assertRaisesRegex(RuntimeError,'test_in_validation'):validation_hook.cohort()
 def test_public_rope_dtype_compatibility(self):
  from common import install_rope_dtype_compatibility
  from dinov3.models import build_model_from_cfg
  install_rope_dtype_compatibility()
  with torch.device('meta'):model,_=build_model_from_cfg(configuration(),only_teacher=True)
  self.assertEqual(model.rope_embed.dtype,torch.float32)
 def test_native_teacher_directory_loads_via_unmodified_official_loader(self):
  import torch.distributed as dist
  import torch.distributed.checkpoint as dcp
  from dinov3.checkpointer import init_fsdp_model_from_checkpoint
  with tempfile.TemporaryDirectory() as tmp:
   dist.init_process_group('gloo',init_method='file://'+str(Path(tmp)/'rendezvous'),rank=0,world_size=1)
   try:
    source=torch.nn.ModuleDict(dict(backbone=torch.nn.Linear(3,4),dino_head=torch.nn.Linear(4,5),ibot_head=torch.nn.Linear(4,6)))
    target=torch.nn.ModuleDict(dict(backbone=torch.nn.Linear(3,4),dino_head=torch.nn.Linear(4,5),ibot_head=torch.nn.Linear(4,6)))
    directory=Path(tmp)/'teacher';dcp.save({'iteration':0,'model':source.state_dict()},storage_writer=dcp.FileSystemWriter(directory))
    init_fsdp_model_from_checkpoint(target,str(directory),skip_load_keys=[],keys_not_sharded=[])
    for key,value in source.state_dict().items():self.assertTrue(torch.equal(value,target.state_dict()[key]))
   finally:dist.destroy_process_group()

if __name__=='__main__':unittest.main()
