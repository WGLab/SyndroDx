"""Global face query attends to patient HPO token representations in two blocks."""
import torch
from torch import nn
from common import WORK,load
from model import Alignment

class Fusion(nn.Module):
    def __init__(self,classes,mode):
        super().__init__();self.mode=mode;self.alignment=Alignment()
        s=torch.load(load(WORK/'best_checkpoint.json')['path'],map_location='cpu',weights_only=False)
        self.alignment.load_state_dict(s['model'],strict=True)
        self.alignment.logit_scale.requires_grad_(False)
        self.attention=nn.ModuleList([nn.MultiheadAttention(256,8,dropout=.1,batch_first=True) for _ in range(2)])
        self.norm=nn.ModuleList([nn.LayerNorm(256) for _ in range(2)])
        self.classifier=nn.Sequential(nn.LayerNorm(256),nn.Dropout(.2),nn.Linear(256,classes))
        if mode=='image':
            self.alignment.text.requires_grad_(False);self.alignment.text_projection.requires_grad_(False)
        if mode=='hpo':self.alignment.image_projection.requires_grad_(False)
        if mode!='fusion':self.attention.requires_grad_(False);self.norm.requires_grad_(False)

    def forward(self,features,ids,mask):
        image=self.alignment.image_projection(features.float())
        if self.mode=='image':return self.classifier(image)
        hidden=self.alignment.text(input_ids=ids,attention_mask=mask).last_hidden_state
        tokens=self.alignment.text_projection(hidden)
        if self.mode=='hpo':return self.classifier((tokens*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True))
        query=image.unsqueeze(1)
        for attn,norm in zip(self.attention,self.norm):
            z,_=attn(query,tokens,tokens,key_padding_mask=~mask.bool(),need_weights=False)
            query=norm(query+z)
        return self.classifier(query[:,0])
