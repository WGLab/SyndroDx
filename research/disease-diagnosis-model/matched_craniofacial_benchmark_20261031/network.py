import sys,importlib.util
import torch
from torch import nn
import torch.nn.functional as F
from common import *
ID=PROJECT/'transface_arcface_identity_cpt_20261026'
sys.path[:0]=[str(ID),str(ID/'dependencies'),str(PROJECT/'arcface_cpt_20261026/dependencies')]
from onnx2torch import convert
from model import author_backbone,IdentityHead,fixed_batchnorm

class Encoder(nn.Module):
    def __init__(self,arm,regime,classes):
        super().__init__();self.arm=arm;self.regime=regime
        pin=load(WORK/'initialization_pins.json')[arm]
        assert digest(pin['path'])==pin['sha256']
        if arm=='r50':
            self.body=convert(str(WORK/'r50_initial.onnx'));out=list(self.body.graph.nodes)[-1];bn=out.args[0];gemm=bn.args[0];flat=gemm.args[0]
            if regime=='arcface_full' and hasattr(self.body,'initializers'):
                # onnx2torch stores PReLU slopes as buffers; make them trainable
                # for genuine full-backbone adaptation, preserving their values.
                for name,value in list(self.body.initializers.named_buffers(recurse=False)):
                    if value.is_floating_point() and value.ndim==3 and tuple(value.shape[1:])==(1,1):
                        delattr(self.body.initializers,name);self.body.initializers.register_parameter(name,nn.Parameter(value.clone()))
            self.feature=nn.Sequential(self.body.get_submodule(gemm.target),self.body.get_submodule(bn.target));b,g=bn.target,gemm.target;out.args=(flat,)
            self.body.graph.erase_node(bn);self.body.graph.erase_node(gemm);self.body.graph.lint();self.body.recompile();self.body.delete_submodule(g);self.body.delete_submodule(b)
        else:
            self.body=author_backbone(optimized=True)
            if arm=='transface_chop_cpt':
                s=load(PROJECT/'arcface_transfer_evaluation_20261028/selection.json');assert s['step']==1000 and digest(s['path'])==s['sha256']
                state=torch.load(s['path'],map_location='cpu',weights_only=False)['backbone']
                self.body.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items()},strict=True)
            self.feature=self.body.feature;self.body.feature=nn.Identity()
        self.head=IdentityHead(classes,scale=64.,margin=.5) if regime=='arcface_full' else nn.Sequential(nn.Dropout(.5),nn.Linear(512,classes))
        if regime=='wce_feature_only':self.body.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode)
        if self.regime=='wce_feature_only':self.body.eval()
        else:fixed_batchnorm(self)
        return self
    def embedding(self,x):
        if self.regime=='wce_feature_only':
            with torch.no_grad():z=self.body(x) if self.arm=='r50' else self.body.forward_features(x)[0]
        else:z=self.body(x) if self.arm=='r50' else self.body.forward_features(x)[0]
        return self.feature(z)
    def forward(self,x,y=None,margin=True):
        z=self.embedding(x)
        if self.regime=='arcface_full':
            if margin:return self.head(z,y)
            return 64.*F.linear(F.normalize(z.float(),dim=1),F.normalize(self.head.weight.float(),dim=1))
        return self.head(z)
    @torch.no_grad()
    def embed(self,x):return F.normalize(self.embedding(x).float(),dim=-1)
