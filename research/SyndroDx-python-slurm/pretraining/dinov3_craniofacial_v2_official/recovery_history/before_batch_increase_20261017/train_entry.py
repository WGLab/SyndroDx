"""Thin local-data/verification adapters around the UNCHANGED upstream trainer."""
import sys,os,time,math
from common import ROOT,V1,UPSTREAM,load,save,now,install_rope_dtype_compatibility
sys.path[:0]=[str(UPSTREAM),str(V1)]
import torch
import torch.distributed as dist
import dinov3.train.train as official
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dataset import make_dataset as local_dataset
from torch.utils.data import Dataset
from PIL import Image

class SyntheticDataset(Dataset):
 def __init__(self,transform,target_transform):self.transform=transform;self.target_transform=target_transform
 def __len__(self):return 8192
 def __getitem__(self,index):
  image=Image.new('RGB',(192,224),(index%255,(index*7)%255,(index*19)%255))
  return self.transform(image),self.target_transform(0) if self.target_transform else 0

def registered_dataset(*,dataset_str,transform,target_transform=None,**kwargs):
 if os.environ.get('CPT_PHASE')=='capacity':return SyntheticDataset(transform,target_transform)
 return local_dataset(dataset_str=dataset_str,transform=transform,target_transform=target_transform,**kwargs)

original_init=SSLMetaArch.init_weights

def verified_init(self):
 original_init(self)
 reference=torch.load(ROOT/'initialization/public_synthetic_reference.pt',map_location='cpu',weights_only=True)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  actual=self.teacher.backbone(reference['input'].cuda(),is_training=True)['x_norm_clstoken'].float()
 score=torch.nn.functional.cosine_similarity(actual,reference['expected_cls'].cuda()).min()
 dist.all_reduce(score,op=dist.ReduceOp.MIN)
 if not bool(torch.isfinite(score)) or float(score)<.999:raise RuntimeError('public_backbone_initialization_equivalence_failed')
 if dist.get_rank()==0:save(ROOT/'initialization/gpu_equivalence.json',dict(state='passed',world_size=dist.get_world_size(),public_synthetic_input_only=True,minimum_cosine=float(score),checked_at_utc=now()))
 torch.cuda.reset_peak_memory_stats()


def main():
 os.umask(0o077);torch.set_num_threads(4);install_rope_dtype_compatibility()
 official.make_dataset=registered_dataset
 SSLMetaArch.init_weights=verified_init
 if os.environ.get('CPT_PHASE')!='capacity':
  from validation_hook import install
  install(official)
 started=time.monotonic();success=False;failure=None
 try:official.main();success=True
 except Exception as exc:
  failure=exc;raise
 finally:
  if os.environ.get('CPT_PHASE')=='capacity' and torch.cuda.is_initialized():
   folder=ROOT/'capacity'/('batch_'+os.environ['CPT_BATCH'])
   save(folder/('rank_'+os.environ.get('RANK','unknown')+'.json'),dict(state='passed' if success else 'failed',rank=int(os.environ.get('RANK','-1')),peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-started,clinical_data_accessed=False,out_of_memory=isinstance(failure,torch.cuda.OutOfMemoryError),error_type=type(failure).__name__ if failure else None,completed_at_utc=now()))

if __name__=='__main__':main()
