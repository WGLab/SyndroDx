"""WCE stage starts from disease ArcFace weights, then freezes the face body."""
import sys,torch
from torch import nn
import torch.nn.functional as F
from common import PROJECT
sys.path.insert(0,str(PROJECT/'transface_arcface_identity_cpt_20261026'))
sys.path.insert(0,str(PROJECT/'transface_arcface_identity_cpt_20261026/dependencies'))
from model import backbone

class Model(nn.Module):
    def __init__(self,objective,p):
        super().__init__();self.backbone=backbone()
        self.head=nn.Sequential(nn.Dropout(.5),nn.Linear(512,p['num_identity_classes']))
    def configure_feature_only(self):
        self.backbone.requires_grad_(False)
        assert isinstance(self.backbone.encoder.feature[-1],nn.BatchNorm1d)
        self.backbone.encoder.feature[-1]=nn.Identity()
        self.backbone.encoder.feature.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.backbone.encoder.feature.train(mode);return self
    def forward(self,x,y=None):return self.head(self.backbone(x))
    @torch.no_grad()
    def embed(self,x):return F.normalize(self.backbone(x).float(),dim=-1)
