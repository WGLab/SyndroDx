"""GMDB records stay local; checkpoint selection sees development patients only."""
import collections
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from common import WORK,load,digest

def verify_data():
    receipt=load(WORK/'data_receipt.json')
    if receipt.get('state')!='complete' or not receipt.get('patient_disjoint'):
        raise RuntimeError('patient_disjoint_gmdb_release_required')
    for name,sha in receipt['files'].items():
        if digest(WORK/name)!=sha:raise RuntimeError('gmdb_training_release_changed')
    return receipt

class Images(Dataset):
    def __init__(self,indices,train=False):
        self.indices=list(indices)
        self.images=np.load(WORK/'private/images.uint8.npy',mmap_mode='r',allow_pickle=False)
        self.records=load(WORK/'private/records.private.json')['rows']
        operations=[]
        if train:
            operations=[T.RandomResizedCrop(112,scale=(.85,1.),ratio=(.95,1.05),interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(.5),T.RandomGrayscale(.2),
                T.RandomApply([T.ColorJitter(brightness=.15,contrast=.15,saturation=.1,hue=.02)],p=.5)]
        self.transform=T.Compose(operations+[T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])

    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        offset=self.indices[index]
        image=Image.fromarray(self.images[offset])
        try:return self.transform(image),self.records[offset]['class_index'],offset
        finally:image.close()

def macro_hits(scores,targets):
    scores=np.asarray(scores);targets=np.asarray(targets)
    order=np.argsort(-scores,axis=1,kind='stable');classes=np.unique(targets)
    return {f'top{k}':float(np.mean([np.mean(np.any(order[targets==c,:min(k,scores.shape[1])]==c,axis=1)) for c in classes])) for k in (1,5,10)}

def validation_metrics(scores,offsets,records):
    targets=np.asarray([records[i]['class_index'] for i in offsets])
    groups=collections.defaultdict(list)
    for j,i in enumerate(offsets):groups[records[i]['subject']].append(j)
    patient_scores=[];patient_targets=[]
    for positions in groups.values():
        labels=set(int(targets[j]) for j in positions)
        if len(labels)!=1:raise RuntimeError('ambiguous_development_patient_diagnosis')
        patient_scores.append(scores[positions].mean(0));patient_targets.append(next(iter(labels)))
    return dict(image_macro=macro_hits(scores,targets),patient_macro=macro_hits(patient_scores,patient_targets),
        images=len(offsets),patients=len(groups))
