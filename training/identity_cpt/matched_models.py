import copy, importlib.util
import torch
import torch.nn.functional as F
from torch import nn
from matched_runtime import *
from model import backbone, IdentityHead, fixed_batchnorm

class Model(nn.Module):
    def __init__(self,objective,p):
        super().__init__();assert objective=='arcface_identity';self.backbone=backbone();self.objective=objective
        self.head=IdentityHead(p['num_identity_classes'],scale=p['arcface_scale'],margin=p['arcface_margin'])
    def train(self,mode=True):
        super().train(mode);fixed_batchnorm(self.backbone);return self
    def forward(self,x,y=None):
        f=self.backbone(x)
        return self.head(f,y) if self.objective=='arcface_identity' else self.head(f)
    @torch.no_grad()
    def embed(self,x):return F.normalize(self.backbone(x).float(),dim=-1)
