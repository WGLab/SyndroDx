"""Public-weight structure test; no dataset is opened."""
from pathlib import Path
from runtime import ROOT,save,now
from configs import config
from weights import OfficialBackbone
import torch
from dinov3.models import build_model_from_cfg
cfg=config()
with torch.device('meta'):model,_=build_model_from_cfg(cfg,only_teacher=True)
OfficialBackbone(ROOT/'assets/dinov3-vit7b16').load(model,shape_only=True)
save(ROOT/'assets/native_weight_validation.json',{'state':'native_weight_shapes_and_full_key_coverage_passed','clinical_data_accessed':False,'completed_at_utc':now()})
