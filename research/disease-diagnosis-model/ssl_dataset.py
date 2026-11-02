#!/usr/bin/env python
"""DINO multi-crop dataset with session-aware positive sampling.

The domain-specific piece: 72% of these images have another photo of the same
patient taken on the same day. Using that as the second global view gives real
pose / lighting / expression / gaze variation instead of a synthetic colour jitter
of the same pixels -- the multi-instance trick from the medical-imaging SSL
literature, which reliably beats single-image augmentation.

Hard constraint: positives are drawn ONLY from the same (mrn, date). 4,887
patients in this cohort are imaged across >=2 years, so pairing across dates
would pair pre-operative with post-operative morphology and teach the encoder to
discard exactly the signal we want it to keep.
"""
import csv, collections, random, os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T
from expert_reservation import filter_reserved

csv.field_size_limit(10**9)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    def __init__(self, p=0.5, lo=0.1, hi=2.0):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(self.lo, self.hi)))


class Solarize:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


class DinoCrops:
    """2 global crops at `gsize`, n_local crops at `lsize`.

    Geometric augmentation is deliberately mild: no horizontal flip, and a high
    minimum crop scale. Craniofacial diagnosis is a left/right-asymmetry and
    absolute-proportion task -- hemifacial microsomia, plagiocephaly and facial
    palsy are *defined* by asymmetry, so a random flip would destroy label
    information, and aggressive scale jitter destroys the proportion cues.
    """

    def __init__(self, gsize=224, lsize=98, n_local=6,
                 gscale=(0.45, 1.0), lscale=(0.15, 0.45)):
        self.n_local = n_local
        norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        colour = T.Compose([
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ])
        self.g1 = T.Compose([
            T.RandomResizedCrop(gsize, scale=gscale, interpolation=T.InterpolationMode.BICUBIC),
            colour, GaussianBlur(p=1.0), norm])
        self.g2 = T.Compose([
            T.RandomResizedCrop(gsize, scale=gscale, interpolation=T.InterpolationMode.BICUBIC),
            colour, GaussianBlur(p=0.1), Solarize(0.2), norm])
        self.l = T.Compose([
            T.RandomResizedCrop(lsize, scale=lscale, interpolation=T.InterpolationMode.BICUBIC),
            colour, GaussianBlur(p=0.5), norm])

    def __call__(self, img_a, img_b):
        """img_a, img_b may be the same PIL image (no session partner available)
        or two photos from the same session."""
        crops = [self.g1(img_a), self.g2(img_b)]
        for i in range(self.n_local):
            crops.append(self.l(img_a if i % 2 == 0 else img_b))
        return crops


class CraniofacialSSL(Dataset):
    def __init__(self, labels_tsv, gsize=224, lsize=98, n_local=6,
                 session_pair_prob=0.5, exclude_split=None, resized_manifest=None):
        rows = list(csv.DictReader(open(labels_tsv, newline=''), delimiter='\t'))
        rows, reserved_count = filter_reserved(rows)
        print(f"[dataset] excluded {reserved_count:,} expert-reserved/identity-linked rows", flush=True)
        if exclude_split:
            # held-out evaluation patients can be kept out of pretraining entirely
            keep = set(exclude_split)
            rows = [r for r in rows if r['split'] not in keep]
        # Prefer the pre-resized 384px copies. The originals are median
        # 1403x1484 (96 GB total); decoding those to make a 224px crop is what
        # makes the loader the bottleneck rather than the GPU.
        resized = {}
        if resized_manifest and os.path.exists(resized_manifest):
            for r in csv.DictReader(open(resized_manifest, newline=''), delimiter='\t'):
                resized[r['img_id']] = r['resized_path']
        self.paths = [resized.get(r['img_id'], r['cropping_path']) for r in rows]
        n_res = sum(1 for i, r in enumerate(rows) if self.paths[i] != r['cropping_path'])
        print(f"[dataset] using pre-resized copies for {n_res:,}/{len(rows):,} images", flush=True)
        self.mrn = [r['mrn'] for r in rows]
        # session index -> list of row positions sharing (mrn, date)
        sess = collections.defaultdict(list)
        for i, r in enumerate(rows):
            if r['mrn'] and r['date_taken']:
                sess[(r['mrn'], r['date_taken'])].append(i)
        self.partners = [[] for _ in rows]
        for members in sess.values():
            if len(members) < 2:
                continue
            for i in members:
                self.partners[i] = [j for j in members if j != i]
        self.session_pair_prob = session_pair_prob
        self.crops = DinoCrops(gsize, lsize, n_local)
        n_with = sum(1 for p in self.partners if p)
        print(f"[dataset] {len(rows):,} images, {len(set(self.mrn)):,} patients, "
              f"{n_with:,} ({100*n_with/len(rows):.1f}%) have a same-session partner, "
              f"session_pair_prob={session_pair_prob}", flush=True)

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        return Image.open(self.paths[i]).convert('RGB')

    def __getitem__(self, i):
        try:
            a = self._load(i)
        except Exception:
            a = Image.new('RGB', (224, 224))
        b = a
        if self.partners[i] and random.random() < self.session_pair_prob:
            j = random.choice(self.partners[i])
            try:
                b = self._load(j)
            except Exception:
                b = a
        return self.crops(a, b)


def collate(batch):
    """batch: list of per-sample crop lists -> list of stacked tensors by crop id.
    Crops of equal resolution are grouped so the backbone can process each
    resolution in a single forward."""
    n = len(batch[0])
    return [torch.stack([b[k] for b in batch]) for k in range(n)]
