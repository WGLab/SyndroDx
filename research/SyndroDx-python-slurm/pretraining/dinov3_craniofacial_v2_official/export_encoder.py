"""CPU-only teacher backbone export after official training, preserving precision."""
import sys,math,os
from pathlib import Path
from common import ROOT,V1,load,save,now,notify
sys.path.insert(0,str(V1))
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from omegaconf import OmegaConf
from streaming_encoder import StreamingEncoderWriter

def main():
 torch.set_num_threads(4);plan=load(ROOT/'run_plan.json');step=plan['total_steps']-1
 source=ROOT/'training/eval'/f'training_{step}'/'sharded_teacher_checkpoint'
 reader=dcp.FileSystemReader(source);metadata=reader.read_metadata();prefix='model.backbone.'
 state={name:torch.empty(tuple(item.size),dtype=item.properties.dtype) for name,item in metadata.state_dict_metadata.items() if name.startswith(prefix) and hasattr(item,'size')}
 if not state:raise RuntimeError('final_teacher_backbone_not_found')
 dcp.load(state,storage_reader=reader,planner=DefaultLoadPlanner(allow_partial_load=True))
 local={name[len(prefix):]:tensor for name,tensor in state.items()}
 folder=ROOT/'encoder';folder.mkdir(exist_ok=True);temporary=folder/'craniofacial_encoder.safetensors.tmp'
 writer=StreamingEncoderWriter(temporary,local)
 try:
  for name,value in local.items():
   if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):raise RuntimeError('nonfinite_export_tensor')
   writer.write(name,value)
  writer.close()
 except Exception:writer.abort();raise
 temporary.replace(folder/'craniofacial_encoder.safetensors')
 cfg=OmegaConf.load(ROOT/'training/config.yaml');OmegaConf.save(cfg,folder/'training_config.yaml')
 save(folder/'preprocessing.json',dict(mode='RGB',longest_edge=256,padding='aspect-preserving outside original crop only',no_background_removal=True,source='approved native crops; evaluation preprocessing unchanged from original control'))
 save(folder/'encoder_export.json',dict(state='encoder_export_complete',architecture='vit_7b',teacher_EMA_exported=True,source_teacher_checkpoint=str(source),final_validation=load(ROOT/'validation/latest.json'),clinical_superiority_not_established=True,completed_at_utc=now()))
 notify('encoder export complete','Official-code craniofacial encoder exported','Official-code training, scheduled validation and local teacher-encoder export are complete. Review the local validation reports before claiming improvement. Clinical records, images, embeddings and model weights remain on HPC.')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'encoder_export_failure.json',dict(state='failed',error_type=type(exc).__name__,updated_at_utc=now()))
  notify('encoder export failure','Official-code encoder export requires attention','The local CPU export did not complete. All original checkpoints are preserved and private diagnostics remain on HPC.')
  raise SystemExit(1)
