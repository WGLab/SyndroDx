"""Validate FSDP's actual collective primitives across independent Slurm jobs."""
import argparse,datetime,os,socket,time
from pathlib import Path
import torch
import torch.distributed as dist
from runtime import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.cuda.set_device(0);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
rank=dist.get_rank();world=dist.get_world_size();size=8*1024*1024
small=torch.empty(size,device='cuda');large=torch.empty(size*world,device='cuda');reduced=torch.empty_like(small)
start=time.monotonic()
for i in range(12):
 small.fill_(rank+1);dist.all_reduce(small)
 if not bool((small==world*(world+1)//2).all()):raise RuntimeError('all_reduce_mismatch')
 small.fill_(rank+1);dist.all_gather_into_tensor(large,small)
 for j in range(world):
  if not bool((large[j*size:(j+1)*size]==j+1).all()):raise RuntimeError('all_gather_mismatch')
 large.fill_(rank+1);dist.reduce_scatter_tensor(reduced,large)
 if not bool((reduced==world*(world+1)//2).all()):raise RuntimeError('reduce_scatter_mismatch')
torch.cuda.synchronize()
save(Path(a.output)/f'rank_{rank:02d}.json',{'state':'passed','rank':rank,'world_size':world,'host':socket.gethostname(),'elapsed_seconds':time.monotonic()-start,'collectives':['all_reduce','all_gather','reduce_scatter'],'clinical_data_accessed':False})
dist.barrier();dist.destroy_process_group()
