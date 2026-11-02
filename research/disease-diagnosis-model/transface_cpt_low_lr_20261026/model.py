"""Author TransFace-L weights with equivalent SDPA and a training-only DINO head."""
import types
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from control import WEIGHTS
from vendor.transface_vit_original import VisionTransformer
from vendor.dino_head import DINOHead

def sdpa_forward(self,x):
    batch,tokens,width=x.shape
    qkv=self.qkv(x).reshape(batch,tokens,3,self.num_heads,width//self.num_heads).permute(2,0,3,1,4)
    x=F.scaled_dot_product_attention(qkv[0],qkv[1],qkv[2],
        dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
    x=x.transpose(1,2).reshape(batch,tokens,width)
    return self.proj_drop(self.proj(x))

def block_forward(self,x):
    x=x+self.drop_path(self.attn(self.norm1(x)))
    return x+self.drop_path(self.mlp(self.norm2(x)))

def features_forward(self,x):
    x=self.pos_drop(self.patch_embed(x)+self.pos_embed)
    # No stochastic patch masking in clinical CPT; face morphology stays visible.
    for block in self.blocks:
        x=checkpoint(block,x,use_reentrant=False) if self.training else block(x)
    x=self.norm(x.float())
    weights=self.senet(x.flatten(1))
    return (x*weights.unsqueeze(-1)).flatten(1),weights.softmax(dim=1)

def author_backbone(optimized=True):
    model=VisionTransformer(img_size=112,patch_size=9,num_classes=512,embed_dim=768,depth=24,
        num_heads=8,drop_path_rate=.05,norm_layer='ln',mask_ratio=0.,using_checkpoint=False)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    del state
    model.mask_token.requires_grad_(False)
    if optimized:
        for block in model.blocks:
            block.attn.forward=types.MethodType(sdpa_forward,block.attn)
            block.forward=types.MethodType(block_forward,block)
        model.forward_features=types.MethodType(features_forward,model)
    return model

class Features(nn.Module):
    def __init__(self,optimized=True):
        super().__init__();self.encoder=author_backbone(optimized)
    def forward(self,x):
        return self.encoder(x)[0]

def backbone():
    return Features()

def fixed_batchnorm(model):
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()

class Encoder(nn.Module):
    def __init__(self,output_dim=65536):
        super().__init__();self.backbone=backbone()
        self.head=DINOHead(512,output_dim,use_bn=False,norm_last_layer=True)
    def train(self,mode=True):
        super().train(mode);fixed_batchnorm(self.backbone);return self
    def forward(self,images):
        return self.head(self.backbone(images))
    @torch.no_grad()
    def embed(self,images):
        return F.normalize(self.backbone(images).float(),dim=-1)

@torch.no_grad()
def update_teacher(student,teacher,momentum):
    sp,tp=dict(student.named_parameters()),dict(teacher.named_parameters())
    if sp.keys()!=tp.keys():raise RuntimeError('teacher_parameter_mismatch')
    for key in sp:
        tp[key].mul_(momentum).add_(sp[key].detach(),alpha=1-momentum)
    sb,tb=dict(student.named_buffers()),dict(teacher.named_buffers())
    if sb.keys()!=tb.keys():raise RuntimeError('teacher_buffer_mismatch')
    for key in sb:tb[key].copy_(sb[key])
