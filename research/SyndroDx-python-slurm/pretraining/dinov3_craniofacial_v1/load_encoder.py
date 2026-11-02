"""Load the private exported craniofacial encoder locally with no network calls."""
from pathlib import Path
import json,torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from runtime import ROOT
from dinov3.models import build_model_from_cfg

def load_encoder(directory=None,device='cuda'):
 directory=Path(directory or ROOT/'encoder')
 cfg=OmegaConf.load(directory/'training_config.yaml')
 with torch.device('meta'):model,_=build_model_from_cfg(cfg,only_teacher=True)
 model.rope_embed.dtype=torch.float32;model.rope_embed.periods=model.rope_embed.periods.to(torch.float32)
 model.to_empty(device=device)
 model.load_state_dict(load_file(str(directory/'craniofacial_encoder.safetensors'),device='cpu'),strict=True)
 return model.eval()
