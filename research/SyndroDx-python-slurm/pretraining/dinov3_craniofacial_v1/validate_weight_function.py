"""Compare official HF and native Meta features on a synthetic tensor only."""
import datetime,json,os
from pathlib import Path
import torch
from transformers import AutoModel
from runtime import ROOT,save,now
from configs import config
from weights import OfficialBackbone
from dinov3.models import build_model_from_cfg
if torch.cuda.device_count()!=1:raise RuntimeError('single_allocated_GPU_required')
torch.cuda.set_device(0);torch.set_num_threads(4);torch.manual_seed(20261015)
cfg=config()
with torch.device('meta'):native,_=build_model_from_cfg(cfg,only_teacher=True)
native.to_empty(device='cuda');native.rope_embed.dtype=torch.float32;native.init_weights();OfficialBackbone(ROOT/'assets/dinov3-vit7b16').load(native);native.eval()
hf=AutoModel.from_pretrained(str(ROOT/'assets/dinov3-vit7b16'),local_files_only=True,dtype=torch.float32,device_map={'':0},low_cpu_mem_usage=True).eval()
image=torch.randn(1,3,224,224,device='cuda')
with torch.inference_mode():
 meta=native(image,is_training=True);other=hf(pixel_values=image).last_hidden_state
 cls=torch.nn.functional.cosine_similarity(meta['x_norm_clstoken'].float(),other[:,0].float()).min().item()
 patch=torch.nn.functional.cosine_similarity(meta['x_norm_patchtokens'].float(),other[:,5:].float(),dim=-1).min().item()
if cls<.9999 or patch<.9999:
 save(ROOT/'assets/native_weight_function_diagnostics.json',{'synthetic_CLS_cosine':cls,'synthetic_patch_min_cosine':patch,'clinical_data_accessed':False})
 raise RuntimeError('native_and_HF_feature_equivalence_failed')
save(ROOT/'assets/native_weight_function_validation.json',{'state':'passed','synthetic_CLS_cosine':cls,'synthetic_patch_min_cosine':patch,'clinical_data_accessed':False,'completed_at_utc':now()})
