"""Clinical-only adapter. Reference manifests and images are never opened."""
import collections
import csv
import hashlib
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from runtime import ROOT,digest,load


class ClinicalFaces(Dataset):
    def __init__(self,transform,target_transform=None):
        release=load(ROOT/'dataset_release/release.json')
        if (release.get('state')!='ssl_data_ready' or not release.get('patient_split_disjoint')
                or not release.get('protected_pools_excluded')):
            raise RuntimeError('released_patient_isolated_dataset_required')
        self.transform,self.target_transform=transform,target_transform
        self.length=1000000
        spec=release['manifests']['train'];path=Path(spec['path'])
        if digest(path)!=spec['sha256']:raise RuntimeError('released_manifest_digest_mismatch')
        patients=collections.defaultdict(lambda:collections.defaultdict(list))
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream,delimiter='\t'):
                if row.get('training_admitted')!='true' or row.get('split')!='train':
                    raise RuntimeError('nontraining_record_in_training_manifest')
                if row.get('view') not in ('frontal','profile','oblique'):
                    raise RuntimeError('ineligible_view_in_training_manifest')
                patients[row['group_id']][row['view']].append(row['crop_path'])
        self.groups=tuple(tuple(tuple(patients[p][v]) for v in sorted(patients[p])) for p in sorted(patients))
        if not self.groups:raise RuntimeError('empty_clinical_training_source')
        self.image_count=sum(len(images) for views in self.groups for images in views)

    @staticmethod
    def choice(index,salt,n):
        return int.from_bytes(hashlib.blake2b((str(index)+':'+salt).encode(),digest_size=8).digest(),'big')%n

    def __len__(self):return self.length

    def selected_path(self,index):
        views=self.groups[self.choice(index,'patient',len(self.groups))]
        images=views[self.choice(index,'view',len(views))]
        return images[self.choice(index,'photo',len(images))]

    def __getitem__(self,index):
        try:
            with Image.open(self.selected_path(index)) as original:
                image=original.convert('RGB')
            result=self.transform(image)
            image.close()
        except Exception:
            raise RuntimeError('released_clinical_image_could_not_be_decoded') from None
        return result,self.target_transform(0) if self.target_transform else 0


def make_dataset(*,dataset_str,transform,target_transform=None,**kwargs):
    if dataset_str!='released_local_manifests':raise RuntimeError('unapproved_dataset_identifier')
    return ClinicalFaces(transform,target_transform)

