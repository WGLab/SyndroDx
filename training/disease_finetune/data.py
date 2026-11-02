import sys,random,hashlib
import numpy as np,torch
from torch.utils.data import Dataset
from common import *
sys.path.insert(0,os.environ['GESTALTMATCHER_SOURCE'])
from lib.datasets.gestalt_matcher_dataset import GestaltMatcherDataset

class Images(Dataset):
    def __init__(self,records,ids,seed=0,epoch=0,train=False):
        self.records,self.ids,self.seed,self.epoch=records,ids,seed,epoch
        self.arr=np.load(WORK/'private/images.uint8.npy',mmap_mode='r',allow_pickle=False)
        self.upstream=GestaltMatcherDataset.__new__(GestaltMatcherDataset);self.upstream.augment=train;self.upstream.target_size=112;self.upstream.aspect_ratio=False;self.upstream.in_channels=3
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        index=self.ids[i];r=self.records[index]
        # Draws depend on image/epoch/seed, independent of model and worker order.
        seed=int.from_bytes(hashlib.blake2b(f"{self.seed}/{self.epoch}/{r['image_id']}".encode(),digest_size=4).digest(),'little')
        ts=torch.get_rng_state();ns=np.random.get_state();rs=random.getstate()
        try:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            x=self.upstream.preprocess(np.array(self.arr[r['array_index']],copy=True))
        finally:torch.set_rng_state(ts);np.random.set_state(ns);random.setstate(rs)
        return x,r['class_index'],index

def verify():
    p=load(WORK/'data_pins.json');assert load(WORK/'audit.json').get('state')=='passed'
    assert digest(WORK/'private/records.private.json')==p['records_sha256']
    assert digest(WORK/'private/blocked_patients.json')==p['blocked_sha256']
    rows=load(WORK/'private/records.private.json')['rows']
    train={r['subject'] for r in rows if r['split']=='train'};val={r['subject'] for r in rows if r['split']=='validation'};assert not train&val
    return rows

def classification(scores,offsets,records):
    groups={}
    for v,i in zip(scores,offsets):groups.setdefault(records[i]['subject'],[]).append((v,records[i]['class_index']))
    values=[];y=[]
    for rows in groups.values():
        assert len({r[1] for r in rows})==1;values.append(np.mean([r[0] for r in rows],axis=0));y.append(rows[0][1])
    order=np.argsort(-np.array(values),axis=1,kind='stable');y=np.array(y)
    return {f'top{k}':float(np.mean([np.mean(np.any(order[y==c,:min(k,order.shape[1])]==c,axis=1)) for c in np.unique(y)])) for k in [1,5,10]}
