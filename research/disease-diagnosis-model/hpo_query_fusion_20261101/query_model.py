"""Frozen specialist branches with trainable phenotype-conditioned queries."""
import torch
from torch import nn

class QueryLayer(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.hpo=nn.MultiheadAttention(width,8,dropout=.1,batch_first=True)
        self.face=nn.MultiheadAttention(width,8,dropout=.1,batch_first=True)
        self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width);self.n3=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Dropout(.1),nn.Linear(width*4,width))
    def forward(self,q,h,hm,x,xm,conditioned):
        if conditioned:
            v,_=self.hpo(q,h,h,key_padding_mask=~hm,need_weights=False)
            q=self.n1(q+v)
        v,_=self.face(q,x,x,key_padding_mask=~xm,need_weights=False)
        q=self.n2(q+v)
        return self.n3(q+self.ff(q))

class QueryFusion(nn.Module):
    def __init__(self,hpo,mode,width=256):
        super().__init__()
        self.hpo=hpo.requires_grad_(False).eval();self.mode=mode
        self.queries=nn.Parameter(torch.randn(1,8,width)*.02)
        self.local=nn.Sequential(nn.Linear(768,width),nn.LayerNorm(width))
        self.global_project=nn.Sequential(nn.Linear(512,width),nn.LayerNorm(width))
        self.layers=nn.ModuleList([QueryLayer(width) for _ in range(2)])
        self.out=nn.Linear(width,width)
        # Exactly the frozen HPO representation at initialization.
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def train(self,mode=True):
        super().train(mode);self.hpo.eval();return self
    def forward(self,features,local,local_mask,ids,states):
        with torch.no_grad():h,ht,hm=self.hpo.phenotype(ids,states)
        if self.mode=='conditional_global':
            x=self.global_project(features.float())[:,None]
            xm=features.norm(dim=1).gt(0)[:,None]
        else:
            x=self.local(local.float());xm=local_mask.bool()
        present=xm.any(1)
        # Provide one masked-out zero placeholder to keep all-missing attention finite.
        xm=xm.clone();xm[~present,0]=True
        x=x.masked_fill(~present[:,None,None],0)
        q=self.queries.expand(len(x),-1,-1)
        for layer in self.layers:
            q=layer(q,ht,hm,x,xm,self.mode!='unconditioned_local')
        z=h+self.out(q.mean(1))*present[:,None]
        return self.hpo.head(z),z
