"""Offline paired GMDB inference; all per-image data and features stay on HPC."""
import json
import os
from pathlib import Path
import sys
WORK=Path(__file__).resolve().parent
TRAIN=WORK.parent/'transface_vitl_cpt_20261026'
SHARED=WORK.parent/'gmdb_checkpoint_eval_20261026'
sys.path.insert(0,str(TRAIN))
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from control import load,save,digest,now
from model import Encoder

def preprocess(image):
    image=cv2.cvtColor(cv2.resize(image,(112,112)),cv2.COLOR_BGR2RGB)
    value=np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)
    value/=255.;value-=.5;value/=.5
    return value

class Images(Dataset):
    def __init__(self,rows,indices):self.rows,self.indices=rows,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        try:
            row=self.rows[self.indices[index]]
            if digest(row['crop_path'])!=row['crop_sha256']:raise RuntimeError()
            image=cv2.imread(row['crop_path'],cv2.IMREAD_COLOR)
            if image is None:raise RuntimeError()
            return torch.from_numpy(preprocess(image))
        except Exception:raise RuntimeError('frozen_gmdb_input_integrity_or_decode_failure') from None

def main():
    os.umask(0o077);cv2.setNumThreads(1);torch.set_num_threads(4);torch.cuda.set_device(0)
    kind=sys.argv[1];rank=int(sys.argv[2]);world=2
    if kind not in ('original','cpt4000') or rank not in (0,1):raise RuntimeError('unexpected_inference_assignment')
    p=load(WORK/'protocol.json')
    if digest(SHARED/'inputs.private.jsonl')!=p['inputs_sha256']:raise RuntimeError('frozen_inputs_changed')
    with (SHARED/'inputs.private.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    indices=list(range(rank,len(rows),world));model=Encoder().cuda().eval()
    if kind=='cpt4000':
        state=torch.load(p['checkpoint_path'],map_location='cpu',mmap=True,weights_only=False)
        if state['step']!=4000:raise RuntimeError('wrong_checkpoint_step')
        model.load_state_dict(state['teacher'],strict=True);del state
    model.requires_grad_(False)
    loader=DataLoader(Images(rows,indices),batch_size=256,num_workers=2,pin_memory=True)
    pieces=[];completed=0
    with torch.inference_mode():
        for images in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                features=model.embed(images.cuda(non_blocking=True)).float()
            if not torch.isfinite(features).all():raise RuntimeError('nonfinite_embedding')
            pieces.append(features.cpu().numpy());completed+=len(images)
            save(WORK/'progress'/f'{kind}_{rank}.json',dict(state='extracting',model=kind,shard=rank,
                completed=completed,total=len(indices),updated_at_utc=now()))
    values=np.concatenate(pieces)
    if values.shape!=(len(indices),512):raise RuntimeError('incorrect_feature_shape')
    path=WORK/'private'/f'{kind}_{rank}.npy';np.save(path,values)
    save(WORK/'progress'/f'{kind}_{rank}.json',dict(state='complete',model=kind,shard=rank,
        completed=completed,total=len(indices),features_sha256=digest(path),updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'failures'/f'{sys.argv[1]}_{sys.argv[2]}.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
