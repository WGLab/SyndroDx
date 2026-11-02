"""Same frozen-v1 initialization and pooling for all unfreezing arms."""
import importlib.util
import sys
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common import *

ID = PROJECT/'transface_arcface_identity_cpt_20261026'
sys.path[:0]=[str(ID),str(ID/'dependencies')]
spec=importlib.util.spec_from_file_location('unfreeze_author_wrapper',ID/'model.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)


class Visual(nn.Module):
    def __init__(self,arm,state):
        super().__init__();self.arm=arm
        self.body=author.VisionTransformer(img_size=112,patch_size=9,num_classes=512,embed_dim=768,depth=24,
            num_heads=8,drop_path_rate=.05,norm_layer='ln',mask_ratio=0.,using_checkpoint=False)
        self.feature=self.body.feature;self.body.feature=nn.Identity()
        self.body.load_state_dict({k.removeprefix('body.'):v for k,v in state.items() if k.startswith('body.')},strict=True)
        self.feature.load_state_dict({k.removeprefix('feature.'):v for k,v in state.items() if k.startswith('feature.')},strict=True)
        for block in self.body.blocks:
            block.attn.forward=types.MethodType(author.sdpa_forward,block.attn)
            block.forward=types.MethodType(author.block_forward,block)
        self.requires_grad_(False)
        if arm=='full':self.requires_grad_(True)
        elif arm=='partial':
            for module in [*self.body.blocks[-4:],self.body.norm,self.body.senet,self.feature]:module.requires_grad_(True)
        elif arm!='frozen':raise ValueError('unknown_unfreezing_arm')
        self.body.mask_token.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        # eval is intentional: no BN-state or stochastic-depth difference between arms.
        super().train(False)
        return self

    def forward(self,images):
        x=self.body.pos_drop(self.body.patch_embed(images)+self.body.pos_embed)
        for block in self.body.blocks:
            needs_graph=torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in block.parameters()))
            x=checkpoint(block,x,use_reentrant=False) if needs_graph else block(x)
        x=self.body.norm(x.float())
        weight=self.body.senet(x.flatten(1))
        return F.normalize(self.feature((x*weight.unsqueeze(-1)).flatten(1)).float(),dim=1)


class JointModel(nn.Module):
    def __init__(self,arm):
        super().__init__();assets=load(WORK/'assets.json')
        visual=torch.load(assets['visual_encoder']['path'],map_location='cpu',mmap=True,weights_only=False)['model']
        self.vision=Visual(arm,visual);del visual
        spec=importlib.util.spec_from_file_location('unfreeze_locked_fusion',assets['code_model.py']['path'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        counts=load(WORK/'data_receipt.json')['model_counts']
        embeddings=torch.from_numpy(np.load(assets['HPO_term_embeddings']['path']))
        self.fusion=module.PhenotypeModel(embeddings,counts['GMDB_classes'],counts['public_training_diseases'],'fusion_selected')
        state=torch.load(assets['reference_11']['path'],map_location='cpu',weights_only=False)['model']
        if state['assertion.weight'].shape[0]==2:
            w=state['assertion.weight'];state['assertion.weight']=torch.cat([w,w.mean(0,keepdim=True)],0)
        self.fusion.load_state_dict(state,strict=True)
        self.profile_loss=module.profile_loss

    def patient_features(self,images,counts,chunk=32):
        zs=[]
        for start in range(0,len(images),chunk):zs.append(self.vision(images[start:start+chunk]))
        z=torch.cat(zs)
        patient=torch.repeat_interleave(torch.arange(len(counts),device=z.device),counts.to(z.device))
        sums=torch.zeros(len(counts),512,device=z.device,dtype=torch.float32).index_add(0,patient,z)
        return F.normalize(sums/counts.to(z.device)[:,None],dim=1)

    def parameter_counts(self):
        return dict(visual_total=sum(p.numel() for p in self.vision.parameters()),
                    visual_trainable=sum(p.numel() for p in self.vision.parameters() if p.requires_grad),
                    fusion_trainable=sum(p.numel() for p in self.fusion.parameters() if p.requires_grad))
