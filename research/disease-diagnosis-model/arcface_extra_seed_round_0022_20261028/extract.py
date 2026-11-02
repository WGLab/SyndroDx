"""Distributed frozen-encoder development feature extraction, HPC-only."""
import os, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from matched_runtime import *
from data import evaluation_cohort,EvaluationImages

class PublicImages(Dataset):
    def __init__(self,indices):
        self.indices=indices
        self.values=np.load(GMDB/'private/images.uint8.npy',mmap_mode='r',allow_pickle=False)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        a=np.asarray(self.values[self.indices[i]])
        if a.shape==(112,112,3):a=a.transpose(2,0,1)
        if a.shape!=(3,112,112) or a.dtype!=np.uint8:raise RuntimeError('gmdb_array_layout_mismatch')
        return torch.from_numpy(a.copy()).float().div_(127.5).sub_(1)

def extract(model,folder,step):
    rank,world=dist.get_rank(),dist.get_world_size()
    folder.mkdir(parents=True,exist_ok=True)
    records,_=evaluation_cohort()
    patients=load(ROOT/'private/gmdb_patients.json')['patients']
    indices=sorted({i for p in patients for i in p['image_indices']})
    indices_shard=indices[rank::world]
    sets={'internal':EvaluationImages(records,list(range(rank,len(records),world))), 'gmdb':PublicImages(indices_shard)}
    mode=model.training;model.eval()
    for name,ds in sets.items():
        values=[]
        with torch.no_grad():
            for x in DataLoader(ds,batch_size=256,num_workers=2,pin_memory=True):
                with torch.autocast('cuda',dtype=torch.bfloat16):f=model.embed(x.cuda(non_blocking=True))
                if not torch.isfinite(f).all():raise FloatingPointError('nonfinite_eval_features')
                values.append(f.cpu().numpy())
        arr=np.concatenate(values) if values else np.empty((0,512),np.float32)
        temp=folder/f'{name}_{rank}.tmp.npy';np.save(temp,arr);temp.replace(folder/f'{name}_{rank}.npy')
    model.train(mode);dist.barrier()
    if rank==0:
        save(folder/'ready.json',dict(state='ready',step=step,world=world,internal_images=len(records),gmdb_images=len(indices),
            protocol_sha256=digest(ROOT/'protocol.json'),updated_at_utc=now()))
    dist.barrier()
