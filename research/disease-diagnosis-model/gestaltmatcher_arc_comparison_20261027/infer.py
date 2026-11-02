"""Four author-style variants per model, saved locally for matched ranking."""
import csv,sys,os
import cv2,numpy as np,torch
import torch.nn.functional as F
from common import *
from disease_model import DiseaseModel,ArcFace

@torch.no_grad()
def main(arm):
    os.umask(0o077);cv2.setNumThreads(1);torch.set_num_threads(4);torch.cuda.set_device(0)
    folder=WORK/'evaluation'/arm;folder.mkdir(parents=True,exist_ok=True)
    if load(folder/'receipt.json').get('state')=='complete':return
    ref=load(SHARED/'protocol.json')
    if digest(SHARED/'inputs.private.jsonl')!=ref['inputs_sha256']:raise RuntimeError('evaluation_inputs_changed')
    with (SHARED/'inputs.private.jsonl').open() as f:rows=[__import__('json').loads(s) for s in f]
    if arm=='unadapted':model=ArcFace('r100').cuda().eval()
    else:
        selected=load(WORK/'cases'/arm/'best.json')
        if digest(selected['path'])!=selected['sha256']:raise RuntimeError('selected_checkpoint_changed')
        model=DiseaseModel(load(WORK/'data_receipt.json')['diseases'],arm,load(WORK/'selected/identity_checkpoint.json'))
        model.load_adaptation(torch.load(selected['path'],map_location='cpu',weights_only=False)['adaptation']);model.cuda().eval()
    model.requires_grad_(False)
    path=folder/'features.private.npy';tmp=path.with_suffix('.tmp')
    values=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(rows),4,512))
    for start in range(0,len(rows),64):
        images=[]
        for r in rows[start:start+64]:
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('crop_changed')
            bgr=cv2.imread(r['crop_path'])
            if bgr is None:raise RuntimeError('crop_decode_failed')
            rgb=cv2.cvtColor(cv2.resize(bgr,(112,112)),cv2.COLOR_BGR2RGB)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray=np.repeat(gray[:,:,None],3,axis=2)
            images.append([rgb,rgb[:,::-1].copy(),gray,gray[:,::-1].copy()])
        x=torch.from_numpy(np.asarray(images)).cuda().flatten(0,1).permute(0,3,1,2).float().div_(127.5).sub_(1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=model.feature(model.forward_features(x)[0]) if arm=='unadapted' else model(x)[1]
        features=F.normalize(features.float(),dim=-1)
        if not torch.isfinite(features).all():raise RuntimeError('nonfinite_test_embedding')
        values[start:start+len(images)]=features.cpu().numpy().reshape(len(images),4,512)
        save(folder/'status.json',dict(state='extracting',images=start+len(images),total=len(rows),updated_at_utc=now()))
    values.flush();del values;tmp.replace(path)
    save(folder/'receipt.json',dict(state='complete',arm=arm,features_sha256=digest(path),inputs_sha256=ref['inputs_sha256'],variants=['RGB','flip','gray','gray_flip'],updated_at_utc=now()))

if __name__=='__main__':main(sys.argv[1])
