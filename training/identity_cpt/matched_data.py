import random
import numpy as np
import torch
from data import ClinicalFaces, OneView

class TwoViews:
    def __init__(self):self.one=OneView()
    def __call__(self,image):return self.one(image),self.one(image)

class MatchedFaces(ClinicalFaces):
    def __init__(self):
        super().__init__();self.transform=TwoViews()
    def __getitem__(self,index):
        # Augmentations depend on global exposure index, never on GPU/worker/arm.
        seed=self.choice(index,'augmentation_20261028',2**31-1)
        ts=torch.get_rng_state();rs=random.getstate();ns=np.random.get_state()
        try:
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            return super().__getitem__(index)
        finally:torch.set_rng_state(ts);random.setstate(rs);np.random.set_state(ns)
