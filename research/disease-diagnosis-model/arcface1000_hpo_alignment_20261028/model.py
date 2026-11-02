"""Frozen TransFace features plus trainable biomedical language alignment."""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from common import WORK

class Alignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.text=AutoModel.from_pretrained(WORK/'assets/biomedbert',local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa')
        # Initially adapt the top four language blocks, retaining pretrained lower-level terminology.
        for p in self.text.parameters():p.requires_grad_(False)
        for layer in self.text.encoder.layer[-4:]:
            for p in layer.parameters():p.requires_grad_(True)
        self.image_projection=nn.Linear(512,256,bias=False)
        self.text_projection=nn.Linear(self.text.config.hidden_size,256,bias=False)
        self.logit_scale=nn.Parameter(torch.tensor(2.65926))

    def forward(self,features,input_ids,attention_mask):
        hidden=self.text(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        image=F.normalize(self.image_projection(features.float()),dim=-1)
        text=F.normalize(self.text_projection(pooled.float()),dim=-1)
        return image,text,self.logit_scale.exp().clamp(max=100)

def targets(hpo,labels):
    intersection=hpo@hpo.T
    union=hpo.sum(1)[:,None]+hpo.sum(1)[None,:]-intersection
    similarity=intersection/union.clamp_min(1)
    positive=torch.where(similarity>=.5,similarity,torch.zeros_like(similarity))
    positive.fill_diagonal_(1)
    # Related patients with uncertain matching semantics are not forced apart.
    neutral=((similarity>0)|(labels[:,None]==labels[None,:]))&(positive==0)
    return positive/positive.sum(1,keepdim=True),neutral

def semantic_loss(image,text,scale,hpo,labels):
    target,neutral=targets(hpo.float(),labels)
    logits=(scale.float()*(image.float()@text.float().T)).masked_fill(neutral,-1e4)
    image_loss=-(target*F.log_softmax(logits,dim=1)).sum(1).mean()
    # The raw similarities are symmetric, hence the reverse target is normalized in the same way.
    text_loss=-(target*F.log_softmax(logits.T,dim=1)).sum(1).mean()
    return (image_loss+text_loss)/2
