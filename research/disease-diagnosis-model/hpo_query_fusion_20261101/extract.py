"""Local-only patch-token extraction; verify original global representations."""
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parent
sys.path.insert(0,str(NEW.parent/'matched_craniofacial_benchmark_20261031'))
from common import *
from network import Encoder
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset,DataLoader

def main():
    os.umask(0o077);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rec=load(NEW/'data_receipt.json');p=load(NEW/'protocol.json')
    assert rec['state']=='passed'
    assert digest(NEW/'private/images.json')==rec['images_sha256']
    assert digest(rec['gmdb_path'])==rec['gmdb_sha256']
    assert digest(p['visual']['path'])==p['visual']['sha256']
    model=Encoder('transface_chop_cpt','arcface_full',load(WORK/'audit.json')['diseases'])
    checkpoint=torch.load(p['visual']['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.cuda().eval().requires_grad_(False)
    rows=load(NEW/'private/images.json')['images']
    class Inputs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('crop_integrity_changed')
            im=cv2.imread(r['crop_path'])
            if im is None:raise RuntimeError('local_decode_failed')
            im=cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB)
            return torch.from_numpy(im.transpose(2,0,1).copy()).float().div_(127.5).sub_(1)
    n=rec['patients_total'];cap=p['max_images'];nt=144;dim=768
    path=NEW/'private/local_tokens.npy';temp=path.with_suffix('.tmp.npy')
    tokens=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float16,shape=(n,cap,nt,dim));tokens[:]=0
    mask=np.zeros((n,cap,nt),bool);sums=np.zeros((n,512),np.float32);counts=np.zeros(n,np.int64)
    captured={}
    hook=model.body.norm.register_forward_hook(lambda module,args,out:captured.update(tokens=out.detach()))
    done=0
    with torch.inference_mode():
        for images in DataLoader(Inputs(),batch_size=128,num_workers=2,pin_memory=True):
            z=model.embed(images.cuda(non_blocking=True)).cpu().numpy()
            patch=captured['tokens'].cpu().numpy()
            assert patch.shape[1:]==(nt,dim) and np.isfinite(patch).all() and np.isfinite(z).all()
            for offset,(v,t) in enumerate(zip(z,patch)):
                row=rows[done+offset];i=row['patient_index'];slot=row['token_slot']
                sums[i]+=v;counts[i]+=1
                if slot>=0:tokens[i,slot]=t;mask[i,slot]=True
            done+=len(z)
            save(NEW/'features/status.json',dict(state='extracting',images_done=done,images_total=len(rows),updated_at_utc=now()))
    hook.remove();tokens.flush();del tokens
    active=counts>0;sums[active]/=counts[active,None];sums[active]/=np.maximum(np.linalg.norm(sums[active],axis=1,keepdims=True),1e-12)
    with np.load(rec['gmdb_path']) as d:reference=d['features'][active]
    error=float(np.max(np.abs(sums[active]-reference)))
    cosine=float(np.min(np.sum(sums[active]*reference,axis=1)))
    if error>2e-5 or cosine<.99999:raise RuntimeError('frozen_native_feature_parity_failed')
    temp.replace(path);np.save(NEW/'private/local_mask.npy',mask)
    assert digest(p['visual']['path'])==p['visual']['sha256']
    save(NEW/'features/receipt.json',dict(state='complete',images=len(rows),patients=int(active.sum()),
        tokens_path=str(path),tokens_sha256=digest(path),mask_sha256=digest(NEW/'private/local_mask.npy'),
        feature_max_abs_error=error,feature_min_cosine=cosine,visual_weights_unchanged=True,
        fp32_extraction=True,cache_dtype='float16',updated_at_utc=now()))
    save(NEW/'features/status.json',dict(state='complete',images_done=done,images_total=len(rows),updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(NEW/'features/failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
