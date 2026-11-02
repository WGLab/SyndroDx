"""Extract true normalized transformer patch states and global face descriptors."""
import importlib.util,os,sys
import numpy as np
import torch
from runtime import *
def vision():
    sys.path.insert(0,str(STAGE1))
    spec=importlib.util.spec_from_file_location('cranio_transface_source',STAGE1/'model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    model=m.backbone();receipt=load(GMDB/'selected/identity_checkpoint.json')
    if receipt['step']!=2500 or digest(receipt['path'])!=receipt['sha256']:raise RuntimeError('vision_checkpoint_changed')
    state=torch.load(receipt['path'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict({k.removeprefix('backbone.'):v for k,v in state['student'].items() if k.startswith('backbone.')},strict=True)
    return model.requires_grad_(False).eval()
def extract(shard,world):
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    rows=load(WORK/'private/patients.json')['patients'];ids=sorted({i for r in rows for i in r['image_indices']})[shard::world]
    images=np.load(GMDB/'private/images.uint8.npy',mmap_mode='r')
    model=vision().cuda();cache={}
    hook=model.encoder.norm.register_forward_hook(lambda mod,inp,out:cache.update(patches=out.detach()))
    patches=[];global_features=[]
    with torch.inference_mode():
        for start in range(0,len(ids),128):
            x=torch.from_numpy(np.array(images[ids[start:start+128]])).cuda().permute(0,3,1,2).float()/127.5-1
            with torch.autocast('cuda',dtype=torch.bfloat16):g=model(x)
            p=cache['patches']
            if p.shape[1:]!=(144,768) or not torch.isfinite(p).all() or not torch.isfinite(g).all():raise RuntimeError('invalid_native_features')
            patches.append(p.cpu().half().numpy());global_features.append(torch.nn.functional.normalize(g.float(),dim=-1).cpu().numpy())
    hook.remove();path=WORK/'features'/f'{shard}.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,indices=np.asarray(ids),patches=np.concatenate(patches),global_features=np.concatenate(global_features))
    tmp.replace(path)
    save(WORK/'features'/f'{shard}.json',dict(state='complete',shard=shard,world=world,images=len(ids),sha256=digest(path),updated_at_utc=now()))
def finalize(world):
    pieces=[]
    for i in range(world):
        p=WORK/'features'/f'{i}.npz';r=load(p.with_suffix('.json'))
        if r.get('state')!='complete' or digest(p)!=r['sha256']:raise RuntimeError('feature_shard_not_sealed')
        pieces.append(np.load(p))
    ids=np.concatenate([p['indices'] for p in pieces]);order=np.argsort(ids)
    if len(ids)!=len(set(ids.tolist())):raise RuntimeError('duplicate_feature_shard')
    rows=load(WORK/'private/patients.json')['patients'];expected=sorted({i for r in rows for i in r['image_indices']})
    if ids[order].tolist()!=expected:raise RuntimeError('feature_coverage_mismatch')
    mapping={int(x):i for i,x in enumerate(ids[order])}
    for r in rows:r['feature_indices']=[mapping[i] for i in r['image_indices']]
    for key in ['patches','global_features']:np.save(WORK/'private'/(key+'.npy'),np.concatenate([p[key] for p in pieces])[order])
    save(WORK/'private/training_index.json',dict(patients=rows))
    save(WORK/'feature_receipt.json',dict(state='complete',images=len(ids),native_patch_shape=[144,768],
        files={n:digest(WORK/n) for n in ['private/training_index.json','private/patches.npy','private/global_features.npy']},updated_at_utc=now()))
if __name__=='__main__':
    if sys.argv[1]=='finalize':finalize(int(sys.argv[2]))
    else:extract(int(sys.argv[1]),int(sys.argv[2]))
