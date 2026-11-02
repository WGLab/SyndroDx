"""Compare selected final teacher tensors to exported tensors locally, read only."""
import sys
sys.path.insert(0,'/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
from runtime import ROOT,save,now
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from safetensors import safe_open

def main():
 torch.set_num_threads(1)
 reader=dcp.FileSystemReader(ROOT/'training/ckpt/2135');metadata=reader.read_metadata()
 keys=['cls_token','storage_tokens','patch_embed.proj.weight','patch_embed.proj.bias','norm.weight','norm.bias','cls_norm.weight','cls_norm.bias','blocks.0.norm1.weight','blocks.39.norm2.weight']
 selected={}
 for k in keys:
  name='model.teacher.backbone.'+k
  item=metadata.state_dict_metadata.get(name)
  if item is not None and hasattr(item,'size'):selected[name]=torch.empty(tuple(item.size),dtype=item.properties.dtype)
 if len(selected)<6:raise RuntimeError('insufficient_teacher_checkpoint_keys')
 dcp.load(selected,storage_reader=reader,planner=DefaultLoadPlanner(allow_partial_load=True))
 checks={}
 with safe_open(ROOT/'encoder/craniofacial_encoder.safetensors',framework='pt',device='cpu') as export:
  for name,reference in selected.items():
   key=name.removeprefix('model.teacher.backbone.');actual=export.get_tensor(key)
   checks[key]=dict(exactly_equal=bool(torch.equal(reference,actual)),finite=bool(torch.isfinite(actual).all()))
 report=dict(state='passed' if all(v['exactly_equal'] and v['finite'] for v in checks.values()) else 'failed',sampled_tensor_count=len(checks),sampled_checks=checks,scope='selected tensors only; not full-model or functional equivalence',clinical_images_accessed=False,tensor_values_exposed=False,completed_at_utc=now())
 save(ROOT/'training/diagnostics/final_export_sample_audit.json',report)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'training/diagnostics/final_export_sample_audit.json',dict(state='failed',error_type=type(exc).__name__,clinical_images_accessed=False,updated_at_utc=now()))
  raise SystemExit(1)
