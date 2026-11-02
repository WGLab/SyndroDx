import sys, importlib.util
import torch
from torch import nn
import torch.nn.functional as F
from common import *
ID = CODE / 'identity_inputs'
sys.path.insert(0, str(ID))
from model import author_backbone, IdentityHead, fixed_batchnorm

class Encoder(nn.Module):

    def __init__(self, arm, regime, classes):
        super().__init__()
        assert arm == 'transface_chop_cpt' and regime == 'arcface_full'
        self.arm = arm
        self.regime = regime
        pin = load(WORK / 'initialization_pins.json')[arm]
        assert digest(pin['path']) == pin['sha256']
        self.body = author_backbone(optimized=True)
        s = load(PROJECT / 'arcface_transfer_evaluation_20261028/selection.json')
        assert s['step'] == 1000 and digest(s['path']) == s['sha256']
        state = torch.load(s['path'], map_location='cpu', weights_only=False)['backbone']
        self.body.load_state_dict({k.removeprefix('encoder.'): v for (k, v) in state.items()}, strict=True)
        self.feature = self.body.feature
        self.body.feature = nn.Identity()
        self.head = IdentityHead(classes, scale=64.0, margin=0.5)

    def train(self, mode=True):
        super().train(mode)
        fixed_batchnorm(self)
        return self

    def embedding(self, x):
        z = self.body.forward_features(x)[0]
        return self.feature(z)

    def forward(self, x, y=None, margin=True):
        z = self.embedding(x)
        if margin:
            return self.head(z, y)
        return 64.0 * F.linear(F.normalize(z.float(), dim=1), F.normalize(self.head.weight.float(), dim=1))

    @torch.no_grad()
    def embed(self, x):
        return F.normalize(self.embedding(x).float(), dim=-1)
