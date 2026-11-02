"""One frozen encoder, canonical RGB pixels, private feature artifacts only."""
import os,sys,json
import numpy as np,torch,cv2
from torch.utils.data import Dataset,DataLoader
from common import *
from network import Encoder

def main():
    os.umask(0o077);name=sys.argv[1];initial=name.startswith('init__')
    case=dict(arm=name.removeprefix('init__'),regime='wce_feature_only') if initial else load(WORK/'cases.json')['cases'][name]
    p=load(WORK/'protocol.json');pins=load(WORK/'data_pins.json')
    assert digest(SHARED/'inputs.private.jsonl')==pins['public_inputs_sha256']
    m=Encoder(case['arm'],case['regime'],load(WORK/'audit.json')['diseases'])
    if not initial:
        source=load(WORK/'cases'/name/'result.json');assert digest(source['path'])==source['sha256']
        state=torch.load(source['path'],map_location='cpu',weights_only=False);assert state['epoch']==50 and state['protocol_sha256']==digest(WORK/'protocol.json')
        m.load_state_dict(state['model'],strict=True);del state
    m=m.cuda().eval().requires_grad_(False);torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()];blocked=set(load(WORK/'private/blocked_patients.json')['subjects'])
    class Inputs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            if r['subject'] in blocked:return torch.zeros(3,112,112)
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('public_crop_integrity_changed')
            im=cv2.imread(r['crop_path'])
            if im is None:raise RuntimeError('local_decode_failure')
            im=cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB)
            return torch.from_numpy(im.transpose(2,0,1).copy()).float().div_(127.5).sub_(1)
    values=[]
    with torch.inference_mode():
        for x in DataLoader(Inputs(),batch_size=128,num_workers=2,pin_memory=True):
            f=m.embed(x.cuda(non_blocking=True))
            if not torch.isfinite(f).all():raise FloatingPointError('nonfinite_features')
            values.append(f.cpu().numpy())
    path=WORK/'private'/f'{name}_features.npy';temp=path.with_suffix('.tmp.npy');np.save(temp,np.concatenate(values));temp.replace(path)
    save(WORK/'inference'/f'{name}.json',dict(state='complete',sha256=digest(path),case=name,precision='FP32 RGB',updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:save(WORK/'inference'/f'{sys.argv[1]}_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
