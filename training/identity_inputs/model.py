"""Author TransFace-L backbone with a new patient-identity ArcFace head."""
import types
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from control import WEIGHTS
from runtime import upstream
VisionTransformer=upstream('transface_author_vit','TRANSFACE_VIT_FILE').VisionTransformer
ArcFace=upstream('transface_author_margin','ARCFACE_LOSS_FILE').ArcFace

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

class IdentityHead(nn.Module):
    def __init__(self, num_classes, feature_dim=512, scale=64., margin=.5):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_classes,feature_dim))
        nn.init.normal_(self.weight,std=.01)
        self.margin=ArcFace(s=scale,margin=margin)

    def forward(self,features,labels):
        with torch.autocast(device_type=features.device.type,enabled=False):
            cosine=F.linear(F.normalize(features.float(),dim=1),F.normalize(self.weight.float(),dim=1))
            # Prevent sqrt gradients at exactly +/-1 in the author's ArcFace.
            cosine=cosine.clamp(-1.+1e-7,1.-1e-7)
            return self.margin(cosine,labels.reshape(-1))
