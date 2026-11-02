"""Official parameter grouping and freezing, robust to ONNX operator names."""
import sys,torch
from torch import nn
import torch.nn.functional as F
from common import WORK,PROJECT,load,digest
sys.path.insert(0,str(PROJECT/'arcface_cpt_20261026/dependencies'))
from onnx2torch import convert

class R50(nn.Module):
    def __init__(self,classes,arm):
        super().__init__();p=load(WORK/'protocol.json');path=WORK/'assets/glint360k_r50.onnx'
        assert digest(path)==p['initialization_sha256']
        self.base=convert(str(path));out=list(self.base.graph.nodes)[-1];bn=out.args[0];gemm=bn.args[0];flat=gemm.args[0]
        assert isinstance(self.base.get_submodule(bn.target),nn.BatchNorm1d)
        self.features=nn.Sequential(self.base.get_submodule(gemm.target),self.base.get_submodule(bn.target) if arm=='code_keep_bn' else nn.Identity())
        bname,gname=bn.target,gemm.target;out.args=(flat,)
        self.base.graph.erase_node(bn);self.base.graph.erase_node(gemm);self.base.graph.lint();self.base.recompile()
        self.base.delete_submodule(gname);self.base.delete_submodule(bname);self.base.requires_grad_(False)
        self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(512,classes))
    def forward(self,x):
        # Match upstream: frozen parameters, train-mode BN buffers retained.
        with torch.no_grad():z=self.base(x)
        z=self.features(z);return self.classifier(z),z
    @torch.no_grad()
    def embed(self,x):return F.normalize(self(x)[1].float(),dim=-1)
