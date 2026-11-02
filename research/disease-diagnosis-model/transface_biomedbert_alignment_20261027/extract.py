"""Disjoint frozen-vision shards on the currently allocated GPUs."""
import os
import sys
import numpy as np
import torch
from common import WORK,GMDB_STAGE,load,save,digest,now
from vision import build_vision

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    shard=int(sys.argv[1]);world=int(sys.argv[2])
    receipt=load(WORK/'data_receipt.json')
    if receipt.get('state')!='complete' or digest(WORK/'private/patients.json')!=receipt['patient_file_sha256']:
        raise RuntimeError('paired_data_release_required')
    patients=load(WORK/'private/patients.json')['patients']
    all_indices=sorted({i for p in patients for i in p['image_indices']})
    indices=all_indices[shard::world]
    images=np.load(GMDB_STAGE/'private/images.uint8.npy',mmap_mode='r')
    model=build_vision().cuda().eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            ids=indices[start:start+256]
            x=torch.from_numpy(np.array(images[ids])).cuda().permute(0,3,1,2).float().div_(127.5).sub_(1)
            with torch.autocast('cuda',dtype=torch.bfloat16): feat=model(x)
            feat=torch.nn.functional.normalize(feat.float(),dim=-1)
            if not torch.isfinite(feat).all():raise RuntimeError('nonfinite_visual_feature')
            out.append(feat.cpu().numpy())
            save(WORK/'features'/f'shard_{shard}.status.json',dict(state='extracting',completed=start+len(ids),total=len(indices),updated_at_utc=now()))
    features=np.concatenate(out) if out else np.empty((0,512),dtype=np.float32)
    path=WORK/'features'/f'shard_{shard}.npz'
    with path.with_suffix('.tmp').open('wb') as f:np.savez(f,indices=np.asarray(indices),features=features)
    path.with_suffix('.tmp').replace(path)
    save(WORK/'features'/f'shard_{shard}.status.json',dict(state='complete',world=world,shard=shard,
        samples=len(indices),sha256=digest(path),patient_file_sha256=receipt['patient_file_sha256'],updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'features'/f'shard_{sys.argv[1]}.failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
