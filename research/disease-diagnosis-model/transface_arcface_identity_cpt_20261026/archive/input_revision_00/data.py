"""Patient/view balanced clinical-only sampling; no private records printed."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from PIL import Image
from control import WORK, V1, BASE, load, digest

def cache_index():
    receipt=load(WORK/'cache_receipt.json')
    protocol=load(WORK/'protocol.json')
    if receipt.get('state')!='complete' or receipt.get('index_sha256')!=digest(WORK/'cache/index.private.json'):
        raise RuntimeError('verified_local_cache_required')
    if receipt['train_manifest_sha256']!=protocol['train_manifest']['sha256'] or receipt['validation_inputs_sha256']!=protocol['validation_inputs_sha256']:
        raise RuntimeError('cache_cohort_mismatch')
    return load(WORK/'cache/index.private.json')

def cache_path(sha,kind):
    return WORK/'cache'/kind/sha[:2]/(sha+('.png' if kind=='train' else '.npy'))

def verified_cached_path(path,sha,kind):
    # Safe progressive cache: every miss verifies source bytes before derivation.
    # An item lock prevents the background builder and training workers racing.
    from cache_inputs import materialize
    result=materialize((path,sha,kind))
    if result is None:raise RuntimeError('local_cache_integrity_or_decode_failure')
    return cache_path(sha,kind)

class OneView:
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(112, scale=(.85, 1.), ratio=(.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomApply([T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02)], p=.5),
            T.ToTensor(), T.Normalize((.5,)*3, (.5,)*3)])

    def __call__(self, image):
        # No aggressive local crops, flips, nonrigid warps or background removal.
        return self.transform(image)

def verify_manifest(spec):
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise RuntimeError('released_manifest_modified')
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def validate_split_rows(train, validation, test):
    patients = [{row['group_id'] for row in rows} for rows in (train, validation, test)]
    if any(patients[i] & patients[j] for i in range(3) for j in range(i+1, 3)):
        raise RuntimeError('patient_overlap_in_released_splits')
    pixels = [{r['pixel_sha256'] for r in rows if r.get('pixel_sha256')} for rows in (train, validation, test)]
    if any(pixels[i] & pixels[j] for i in range(3) for j in range(i+1, 3)):
        raise RuntimeError('identical_pixels_overlap_in_released_splits')
    for row in train:
        if row.get('training_admitted') != 'true' or row.get('split') != 'train':
            raise RuntimeError('nontraining_record_admitted')
        if row.get('source') != 'approved_internal' or row.get('view') not in ('frontal', 'profile', 'oblique'):
            raise RuntimeError('unapproved_clinical_source_or_view')
    return dict(train_images=len(train), train_patients=len(patients[0]),
        validation_images=len(validation), validation_patients=len(patients[1]),
        patient_disjoint=True, exact_pixel_disjoint=True)

class ClinicalFaces(Dataset):
    def __init__(self):
        protocol=load(WORK/'protocol.json')
        spec = protocol['train_manifest']
        rows = verify_manifest(spec)
        mapping_path=WORK/protocol['class_map']
        if digest(mapping_path)!=protocol['class_map_sha256']:
            raise RuntimeError('identity_class_mapping_modified')
        mapping=load(mapping_path)
        identities=mapping['patient_groups_in_class_order']
        if identities!=sorted({r['group_id'] for r in rows}) or len(identities)!=protocol['num_classes']:
            raise RuntimeError('identity_classes_do_not_match_training_only_manifest')
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in rows:
            if row['source'] != 'approved_internal' or row['split'] != 'train' or row['training_admitted'] != 'true':
                raise RuntimeError('unapproved_training_record')
            groups[row['group_id']][row['view']].append((row['crop_path'],row['crop_sha256']))
        self.groups = tuple(tuple(tuple(sorted(groups[p][v])) for v in sorted(groups[p])) for p in identities)
        self.transform = OneView()
        self.checked = set()

    def __len__(self):
        return 2**50

    @staticmethod
    def choice(index, salt, count):
        value = hashlib.blake2b(f'{index}:{salt}'.encode(), digest_size=8).digest()
        return int.from_bytes(value, 'big') % count

    def __getitem__(self, index):
        try:
            label=self.choice(index,'patient',len(self.groups))
            views = self.groups[label]
            images = views[self.choice(index, 'view', len(views))]
            path, sha = images[self.choice(index, 'image', len(images))]
            if path not in self.checked:
                verified_cached_path(path,sha,'train')
                self.checked.add(path)
            with Image.open(cache_path(sha,'train')) as original:
                image = original.convert('RGB')
                output = self.transform(image)
                image.close()
            return output,label
        except Exception:
            raise RuntimeError('local_clinical_input_integrity_or_decode_failure') from None

class StreamSampler(Sampler):
    """A single cumulative sample stream, sharded without duplicate indices."""
    def __init__(self, consumed, rank, world, batch):
        self.consumed, self.rank, self.world, self.batch = consumed, rank, world, batch

    def __iter__(self):
        for step in range(10**12):
            start = self.consumed + step*self.world*self.batch + self.rank*self.batch
            yield from range(start, start+self.batch)

    def __len__(self):
        return 10**12*self.batch

class EvaluationImages(Dataset):
    def __init__(self, records, indices):
        self.records, self.indices = records, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        try:
            row = self.records[self.indices[offset]]
            sha=row['crop_sha256'];path=verified_cached_path(row['crop_path'],sha,'eval')
            value=np.load(path,allow_pickle=False)
            if value.shape!=(3,112,112) or value.dtype!=np.float32:
                raise RuntimeError('evaluation_cache_shape_dtype_mismatch')
            return torch.from_numpy(value)
        except Exception:
            raise RuntimeError('local_evaluation_input_integrity_or_decode_failure') from None

def evaluation_cohort():
    folder = BASE/'validation'
    plan = load(folder/'plan.json')
    for name, key in [('inputs.private.jsonl','inputs_sha256'), ('labels.private.json','labels_sha256')]:
        if digest(folder/name) != plan[key]:
            raise RuntimeError('frozen_development_cohort_modified')
    with (folder/'inputs.private.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    if any(r['split'] not in ('train','validation') for r in records):
        raise RuntimeError('test_record_in_development_cohort')
    parts = {s: {r['group_id'] for r in records if r['split']==s} for s in ('train','validation')}
    if parts['train'] & parts['validation']:
        raise RuntimeError('development_patient_overlap')
    return records, load(folder/'labels.private.json')['labels']
