import copy, importlib.util
import torch
import torch.nn.functional as F
from torch import nn
from matched_runtime import SSL
from model import backbone, IdentityHead, fixed_batchnorm
spec=importlib.util.spec_from_file_location('matched_dino_head',SSL/'vendor/dino_head.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Model(nn.Module):
    def __init__(self,objective,p):
        super().__init__();self.backbone=backbone();self.objective=objective
        self.head=IdentityHead(p['num_identity_classes'],scale=p['arcface_scale'],margin=p['arcface_margin']) if objective=='arcface_identity' else mod.DINOHead(512,p['dino_output_dim'],use_bn=False,norm_last_layer=True)
    def train(self,mode=True):
        super().train(mode);fixed_batchnorm(self.backbone);return self
    def forward(self,x,y=None):
        f=self.backbone(x)
        return self.head(f,y) if self.objective=='arcface_identity' else self.head(f)
    @torch.no_grad()
    def embed(self,x):return F.normalize(self.backbone(x).float(),dim=-1)

def dino_loss(student,teacher,center,temp):
    first,second=(student.float()/.1).chunk(2)
    q1,q2=F.softmax((teacher.float()-center)/temp,dim=-1).detach().chunk(2)
    return -.5*((q1*F.log_softmax(second,dim=-1)).sum(1).mean()+(q2*F.log_softmax(first,dim=-1)).sum(1).mean())

@torch.no_grad()
def update_teacher(student,teacher,momentum):
    for (n,p),(m,q) in zip(student.named_parameters(),teacher.named_parameters()):
        assert n==m;q.mul_(momentum).add_(p.detach(),alpha=1-momentum)
    for (n,b),(m,c) in zip(student.named_buffers(),teacher.named_buffers()):
        assert n==m;c.copy_(b)
