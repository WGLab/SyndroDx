"""Full read-only CPU comparison of final teacher backbone and safetensors export."""
import sys,os
sys.path.insert(0,'/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
from runtime import ROOT,save,now
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from safetensors import safe_open

def main():
 torch.set_num_threads(4);path=ROOT/'training/diagnostics/full_export_audit.json'
 save(path,dict(state='loading_teacher_checkpoint_on_CPU',updated_at_utc=now()))
 reader=dcp.FileSystemReader(ROOT/'training/ckpt/2135');meta=reader.read_metadata();prefix='model.teacher.backbone.'
 selected={n:torch.empty(tuple(v.size),dtype=v.properties.dtype) for n,v in meta.state_dict_metadata.items() if n.startswith(prefix) and hasattr(v,'size')}
 if not selected:raise RuntimeError('no_teacher_backbone_state')
 dcp.load(selected,storage_reader=reader,planner=DefaultLoadPlanner(allow_partial_load=True))
 count=0;all_equal=True;all_finite=True
 with safe_open(ROOT/'encoder/craniofacial_encoder.safetensors',framework='pt',device='cpu') as export:
  keys={name[len(prefix):] for name in selected}
  if keys!=set(export.keys()):raise RuntimeError('checkpoint_export_schema_differs')
  for name in sorted(selected):
   reference=selected[name];value=export.get_tensor(name[len(prefix):]);count+=1
   all_equal=all_equal and torch.equal(reference,value)
   if value.dtype.is_floating_point:all_finite=all_finite and bool(torch.isfinite(value).all())
   del value
   if count%100==0:save(path,dict(state='comparing_all_teacher_backbone_tensors',checked_tensors=count,updated_at_utc=now()))
 save(path,dict(state='passed' if all_equal and all_finite else 'failed',all_teacher_backbone_tensors_checked=True,all_exactly_equal=all_equal,all_finite=all_finite,tensors_compared=count,checkpoint_iteration=2135,clinical_images_accessed=False,private_tensor_values_exposed=False,completed_at_utc=now(),scope='checkpoint-to-export integrity; does not establish training correctness or representation quality'))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'training/diagnostics/full_export_audit.json',dict(state='failed',error_type=type(exc).__name__,updated_at_utc=now()))
  raise SystemExit(1)
