"""Stream official HF tensors into sharded Meta DINOv3 backbones, with strict coverage."""
import json,re
from pathlib import Path
import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor,Shard,Replicate

class OfficialBackbone:
 def __init__(self,path):
  self.path=Path(path)
  index=self.path/'model.safetensors.index.json'
  if index.exists():self.index=json.loads(index.read_text())['weight_map']
  else:
   with safe_open(self.path/'model.safetensors',framework='pt',device='cpu') as f:self.index={k:'model.safetensors' for k in f.keys()}
  self.used=set()
 def read(self,key):
  self.used.add(key)
  with safe_open(self.path/self.index[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
 def tensor(self,key):
  basic={'cls_token':'embeddings.cls_token','mask_token':'embeddings.mask_token','storage_tokens':'embeddings.register_tokens','patch_embed.proj.weight':'embeddings.patch_embeddings.weight','patch_embed.proj.bias':'embeddings.patch_embeddings.bias'}
  if key in basic:return self.read(basic[key])
  if key in ['norm.weight','norm.bias','local_cls_norm.weight','local_cls_norm.bias','cls_norm.weight','cls_norm.bias']:return self.read('norm.'+key.rsplit('.',1)[1])
  m=re.fullmatch(r'blocks\.(\d+)\.(.+)',key)
  if not m:raise KeyError('unknown_native_backbone_tensor')
  block,part=m.groups();prefix='layer.'+block+'.'
  if part.startswith('attn.qkv.'):
   suffix=part.rsplit('.',1)[1]
   return torch.cat([self.read(prefix+'attention.'+name+'_proj.'+suffix) for name in ['q','k','v']],dim=0)
  if part.startswith('attn.proj.'):return self.read(prefix+'attention.o_proj.'+part.rsplit('.',1)[1])
  for layer in ['1','2']:
   if part=='ls'+layer+'.gamma':return self.read(prefix+'layer_scale'+layer+'.lambda1')
  if part.startswith(('norm1.','norm2.')):return self.read(prefix+part)
  if part.startswith('mlp.w1.'):return self.read(prefix+'mlp.gate_proj.'+part.rsplit('.',1)[1])
  if part.startswith('mlp.w2.'):return self.read(prefix+'mlp.up_proj.'+part.rsplit('.',1)[1])
  if part.startswith('mlp.w3.'):return self.read(prefix+'mlp.down_proj.'+part.rsplit('.',1)[1])
  if part.startswith('mlp.fc1.'):return self.read(prefix+'mlp.up_proj.'+part.rsplit('.',1)[1])
  if part.startswith('mlp.fc2.'):return self.read(prefix+'mlp.down_proj.'+part.rsplit('.',1)[1])
  raise KeyError('unknown_native_backbone_tensor')
 @torch.no_grad()
 def load(self,backbone,shape_only=False):
  for key,target in backbone.state_dict().items():
   if key.endswith(('rope_embed.periods','qkv.bias_mask')):continue
   value=self.tensor(key)
   if key=='mask_token':value=value.reshape(target.shape)
   if tuple(value.shape)!=tuple(target.shape):raise ValueError('backbone_tensor_shape_mismatch')
   if not shape_only:
    if isinstance(target,DTensor):
     local=value
     for mesh_dim,placement in enumerate(target.placements):
      if isinstance(placement,Shard):
       coordinate=target.device_mesh.get_coordinate()[mesh_dim]
       size=local.shape[placement.dim];chunk=(size+target.device_mesh.size(mesh_dim)-1)//target.device_mesh.size(mesh_dim)
       offset=min(coordinate*chunk,size);local=local.narrow(placement.dim,offset,min(chunk,size-offset))
      elif not isinstance(placement,Replicate):raise ValueError('unsupported_backbone_placement')
     if local.shape!=target.to_local().shape:raise ValueError('backbone_local_shard_shape_mismatch')
     target.to_local().copy_(local)
    else:target.copy_(value)
   del value
  if self.used!=set(self.index):raise ValueError('official_checkpoint_key_coverage_mismatch')
