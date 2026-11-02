"""Exercise the actual training loop and changing-world-size resume on synthetic faces."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from runtime import ROOT
from configs import config
from omegaconf import OmegaConf
import train_rank
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.models import vision_transformer as vits
from dinov3.models.vision_transformer import DinoVisionTransformer

class SyntheticDataset(Dataset):
 def __init__(self,transform):self.transform=transform
 def __len__(self):return 1000
 def __getitem__(self,index):
  # Abstract colored rectangles, not a real or generated human photograph.
  image=Image.new('RGB',(160,192),(index%255,(index*19)%255,(index*7)%255))
  return self.transform(image),0
class SyntheticMetaArch(train_rank.ContinualMetaArch):
 def init_weights(self):SSLMetaArch.init_weights(self)

vits.vit_synthetic_unit=lambda **kwargs:DinoVisionTransformer(embed_dim=64,depth=2,num_heads=4,ffn_ratio=3,**kwargs)
train_rank.ContinualMetaArch=SyntheticMetaArch
train_rank.make_dataset=lambda *,transform,**kw:SyntheticDataset(transform)
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
cfg=config(arch='vit_synthetic_unit',batch=2,steps=4,output=a.output)
cfg.dino.head_n_prototypes=128;cfg.ibot.head_n_prototypes=128
cfg.dino.head_hidden_dim=128;cfg.ibot.head_hidden_dim=128
cfg.dino.head_bottleneck_dim=32;cfg.ibot.head_bottleneck_dim=32
cfg.crops.local_crops_number=2;cfg.train.num_workers=0
cfg.checkpointing.period=2
config_path=Path(a.output)/'smoke_config.yaml';config_path.parent.mkdir(parents=True,exist_ok=True)
# Every rank writes its own identical config, avoiding non-atomic shared writes.
config_path=config_path.with_name('config_'+str(__import__('os').environ['RANK'])+'.yaml');OmegaConf.save(cfg,config_path)
train_rank.run(argparse.Namespace(config=str(config_path),segment_steps=2))
