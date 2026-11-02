"""Reference-style frozen face base with trainable feature projection."""
import sys
import torch
from torch import nn
import torch.nn.functional as F
from common import WORK,STAGE1,load,digest
sys.path.insert(0,str(STAGE1))
from model import author_backbone

class DiseaseModel(nn.Module):
    def __init__(self,classes,arm,selection=None):
        super().__init__()
        source=load(WORK/'model_source_receipt.json')
        if not source:raise RuntimeError('pinned_model_source_required')
        for path,sha in source.items():
            if digest(path)!=sha:raise RuntimeError('model_source_or_weights_changed')
        self.face=author_backbone(optimized=True)
        if arm=='identity_cpt':
            if not selection or digest(selection['path'])!=selection['sha256']:
                raise RuntimeError('verified_selected_identity_checkpoint_required')
            state=torch.load(selection['path'],map_location='cpu',mmap=True,weights_only=False)
            prefix='backbone.encoder.'
            weights={k[len(prefix):]:v for k,v in state['student'].items() if k.startswith(prefix)}
            self.face.load_state_dict(weights,strict=True)
            del state,weights
        elif arm!='original':raise RuntimeError('unknown_comparison_arm')
        self.face.requires_grad_(False)
        self.face.feature.requires_grad_(True)
        self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(512,classes))

    def train(self,mode=True):
        super().train(mode)
        self.face.eval()
        self.face.feature.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self,images):
        with torch.no_grad():features,_=self.face.forward_features(images)
        embedding=self.face.feature(features)
        return self.classifier(embedding),embedding

    @torch.no_grad()
    def embed(self,images):
        return F.normalize(self(images)[1].float(),dim=-1)

    def adaptation_state(self):
        return dict(feature=self.face.feature.state_dict(),classifier=self.classifier.state_dict())

    def load_adaptation(self,state):
        self.face.feature.load_state_dict(state['feature'],strict=True)
        self.classifier.load_state_dict(state['classifier'],strict=True)
