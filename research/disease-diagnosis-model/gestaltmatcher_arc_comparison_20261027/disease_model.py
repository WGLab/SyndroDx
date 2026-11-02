"""Author graph split for R50/R100; shared frozen-base disease adaptation."""
import sys,torch
from torch import nn
import torch.nn.functional as F
from common import *
sys.path.insert(0,str(PROJECT/'arcface_cpt_20261026/dependencies'))
from onnx2torch import convert

class ArcFace(nn.Module):
    def __init__(self,arm):
        super().__init__()
        p=WORK/'assets/glint360k_r50.onnx' if arm=='r50' else PROJECT/'arcface_glint360k_eval_20261026/assets/glint360k_r100.onnx'
        if arm=='r100' and digest(p)!='4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf':raise RuntimeError('official_R100_weight_digest_mismatch')
        self.base=convert(str(p))
        output=list(self.base.graph.nodes)[-1];bnnode=output.args[0]
        if isinstance(bnnode,(tuple,list)):bnnode=bnnode[0]
        gemmnode=bnnode.args[0];flatten=gemmnode.args[0]
        if bnnode.op!='call_module' or gemmnode.op!='call_module':raise RuntimeError('unexpected_feature_graph')
        self.feature=nn.Sequential(self.base.get_submodule(gemmnode.target),self.base.get_submodule(bnnode.target))
        output.args=(flatten,)
        bn_target,gemm_target=bnnode.target,gemmnode.target
        self.base.graph.erase_node(bnnode);self.base.graph.erase_node(gemmnode)
        self.base.graph.lint();self.base.recompile();self.base.delete_submodule(gemm_target);self.base.delete_submodule(bn_target)
    def forward_features(self,x):return self.base(x),None

class DiseaseModel(nn.Module):
    def __init__(self,classes,arm,selection=None):
        super().__init__()
        if arm in ('r50','r100'):self.face=ArcFace(arm)
        elif arm in ('transface','transface_original'):
            import importlib.util
            sys.path.insert(0,str(STAGE1))
            sys.path.insert(0,str(STAGE1/'dependencies'))
            spec=importlib.util.spec_from_file_location('matched_transface',STAGE1/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            self.face=m.author_backbone(optimized=True)
            if arm=='transface':
                if digest(selection['path'])!=selection['sha256']:raise RuntimeError('CPT_checkpoint_changed')
                s=torch.load(selection['path'],map_location='cpu',weights_only=False,mmap=True)
                prefix='backbone.encoder.';self.face.load_state_dict({k[len(prefix):]:v for k,v in s['student'].items() if k.startswith(prefix)},strict=True)
        else:raise ValueError('unknown_benchmark_arm')
        self.face.requires_grad_(False);self.face.feature.requires_grad_(True)
        self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(512,classes))
    def train(self,mode=True):
        super().train(mode);self.face.eval();self.face.feature.train(mode);self.classifier.train(mode);return self
    def forward(self,x):
        with torch.no_grad():features,_=self.face.forward_features(x)
        embedding=self.face.feature(features);return self.classifier(embedding),embedding
    def adaptation_state(self):return dict(feature=self.face.feature.state_dict(),classifier=self.classifier.state_dict())
    def load_adaptation(self,s):self.face.feature.load_state_dict(s['feature']);self.classifier.load_state_dict(s['classifier'])
    @torch.no_grad()
    def embed(self,x):return F.normalize(self(x)[1].float(),dim=-1)
