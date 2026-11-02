"""Patient-grouped images and HPO; media stay local and are never logged."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from common import *


def arrays():
    r=load(WORK/'data_receipt.json')
    assert r['state']=='passed' and digest(r['gmdb_path'])==r['gmdb_sha256']
    assert digest(WORK/'private/image_groups.json')==r['image_groups_sha256']
    with np.load(r['gmdb_path']) as d: a={k:np.array(d[k]) for k in d.files}
    return a,load(WORK/'private/image_groups.json')['patients']


class Patients(Dataset):
    def __init__(self,indices,groups):
        self.indices=list(map(int,indices));self.groups=groups
        self.arr=np.load(WORK/'private/images_rgb112.npy',mmap_mode='r')
        self.offsets=load(WORK/'private/patient_image_offsets.json')
    def __len__(self):return len(self.indices)
    def __getitem__(self,n):
        index=self.indices[n];out=[]
        for offset in self.offsets[str(index)]:
            im=np.array(self.arr[offset],copy=True)
            out.append(torch.from_numpy(im.transpose(2,0,1).copy()).float().div_(127.5).sub_(1))
        return index,torch.stack(out)


def collate(rows):
    indices=[i for i,_ in rows];counts=[len(x) for _,x in rows]
    return np.array(indices,np.int64),torch.cat([x for _,x in rows]),torch.tensor(counts,dtype=torch.long)


def macro_hits(scores,truth):
    order=np.argsort(-scores,axis=1,kind='stable')
    hits={k:(order[:,:k]==truth[:,None]).any(1) for k in [1,5]}
    return {f'top{k}':float(np.mean([h[truth==c].mean() for c in np.unique(truth)])) for k,h in hits.items()},hits


def retrieval(z,train,val,labels):
    scores=z[val]@z[train].T;order=np.argsort(-scores,axis=1,kind='stable')
    hit={k:(labels[train][order[:,:k]]==labels[val,None]).any(1) for k in [1,5]}
    return {f'top{k}':float(np.mean([h[labels[val]==c].mean() for c in np.unique(labels[val])])) for k,h in hit.items()},hit


def fewshot(z,train,val,labels):
    result={}
    for k in [1,5]:
        groups={c:train[labels[train]==c] for c in np.unique(labels[train])}
        eligible=sorted(c for c,ix in groups.items() if len(ix)>=k)
        query=val[np.isin(labels[val],eligible)];scores=[]
        if not len(query):result[str(k)]={'eligible_queries':0};continue
        for repeat in range(10):
            rng=np.random.default_rng(9011+k*100+repeat)
            prototypes=np.stack([z[rng.choice(groups[c],k,replace=False)].mean(0) for c in eligible])
            prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-12)
            pred=np.asarray(eligible)[np.argmax(z[query]@prototypes.T,axis=1)]
            scores.append(float(np.mean([(pred[labels[query]==c]==c).mean() for c in np.unique(labels[query])])))
        result[str(k)]=dict(macro_top1_mean=float(np.mean(scores)),macro_top1_SD=float(np.std(scores)),eligible_queries=len(query),eligible_gallery_classes=len(eligible),scope='supervised_disease_retention_not_unseen_transfer')
    return result
