"""Released local manifests with source-, patient-, and view-balanced sampling."""
import collections,csv,hashlib,random
from pathlib import Path
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from runtime import ROOT,RUN,digest,load

class CraniofacialAugmentation:
 def __init__(self,cfg):
  self.global_size=cfg.crops.global_crops_size;self.local_number=cfg.crops.local_crops_number
  self.mean=tuple(cfg.crops.rgb_mean);self.norm=transforms.Normalize(self.mean,tuple(cfg.crops.rgb_std))
  self.photo=transforms.Compose([transforms.RandomApply([transforms.ColorJitter(.12,.12,.08,.02)],p=.7),transforms.RandomApply([transforms.GaussianBlur(5,(.1,1.))],p=.15),transforms.ToTensor(),self.norm])
  self.local=transforms.RandomResizedCrop(cfg.crops.local_crops_size,scale=(.15,.5),ratio=(1.,1.),interpolation=transforms.InterpolationMode.BICUBIC)
 def global_view(self,image):
  # Fit the entire native rectangle without stretching/cutting anatomy. Padding
  # is outside its original pixels; no existing background pixel is removed.
  size=self.global_size;edge=round(size*random.uniform(.9,1.));im=image.copy();im.thumbnail((edge,edge),Image.Resampling.BICUBIC)
  canvas=Image.new('RGB',(size,size),tuple(round(m*255) for m in self.mean))
  canvas.paste(im,(random.randint(0,size-im.width),random.randint(0,size-im.height)))
  return self.photo(canvas)
 def __call__(self,image):return {'global_crops':[self.global_view(image) for _ in range(2)],'local_crops':[self.photo(self.local(image)) for _ in range(self.local_number)]}

class ReleasedFaces(Dataset):
 def __init__(self,transform,target_transform=None):
  release=load(ROOT/'dataset_release/release.json')
  if release.get('state')!='ssl_data_ready' or not release.get('patient_split_disjoint') or not release.get('protected_pools_excluded'):raise RuntimeError('released_patient_isolated_dataset_required')
  self.transform=transform;self.target_transform=target_transform;self.groups={};self.length=1000000
  self.manifests=release['manifests']
  for source,name in [('approved_internal','train'),('fairface_reference','reference_train')]:
   spec=self.manifests[name];path=Path(spec['path'])
   if digest(path)!=spec['sha256']:raise RuntimeError('released_manifest_digest_mismatch')
   bypatient=collections.defaultdict(lambda:collections.defaultdict(list))
   with path.open(newline='') as f:
    for row in csv.DictReader(f,delimiter='\t'):
     if row.get('training_admitted')!='true' or row.get('split')!='train':raise RuntimeError('nontraining_record_in_training_manifest')
     if source=='approved_internal' and row.get('view') not in ['frontal','profile','oblique']:raise RuntimeError('ineligible_view_in_training_manifest')
     bypatient[row['group_id']][row['view']].append(row['crop_path'])
   groups=[]
   for p in sorted(bypatient):groups.append(tuple(tuple(bypatient[p][v]) for v in sorted(bypatient[p])))
   if not groups:raise RuntimeError('empty_required_training_source')
   self.groups[source]=tuple(groups)
 @staticmethod
 def choice(index,salt,n):return int.from_bytes(hashlib.blake2b((str(index)+':'+salt).encode(),digest_size=8).digest(),'big')%n
 def __len__(self):return self.length
 def __getitem__(self,index):
  source='fairface_reference' if index%5==4 else 'approved_internal'
  groups=self.groups[source];views=groups[self.choice(index,'patient',len(groups))]
  pictures=views[self.choice(index,'view',len(views))];path=pictures[self.choice(index,'photo',len(pictures))]
  try:
   with Image.open(path) as original:image=original.convert('RGB')
   result=self.transform(image);image.close()
  except Exception as e:raise RuntimeError('released_training_image_could_not_be_decoded') from None
  return result,self.target_transform(0) if self.target_transform else 0

def make_dataset(*,dataset_str,transform,target_transform=None,**kwargs):
 if dataset_str!='released_local_manifests':raise RuntimeError('unapproved_dataset_identifier')
 return ReleasedFaces(transform,target_transform)
