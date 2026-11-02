"""DINOv3 continuation configuration; effective exposure epochs, no disease labels."""
from pathlib import Path
from omegaconf import OmegaConf
from runtime import ROOT,VENDOR

def config(arch='vit_7b',batch=4,steps=20,output=None):
 from dinov3.configs import get_default_config
 cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(VENDOR/'dinov3/configs/train/dinov3_vit7b16_pretrain.yaml'))
 cfg.student.arch=arch
 cfg.student.fp8_enabled=False
 cfg.student.drop_path_rate=0.1
 cfg.student.untie_global_and_local_cls_norm=True
 cfg.train.batch_size_per_gpu=batch
 cfg.train.num_workers=2
 cfg.train.seed=20261015
 cfg.train.output_dir=str(output or ROOT/'training')
 cfg.train.dataset_path='released_local_manifests'
 cfg.train.compile=False
 cfg.train.checkpointing=True
 cfg.train.checkpointing_full=True
 cfg.train.cache_dataset=True
 cfg.train.OFFICIAL_EPOCH_LENGTH=1
 cfg.optim.epochs=steps
 cfg.optim.scaling_rule='fixed_continual_learning_rate'
 cfg.optim.clip_grad=3.0
 cfg.optim.adamw_beta2=0.999
 cfg.dino.head_n_prototypes=16384
 cfg.dino.head_hidden_dim=2048
 cfg.dino.head_bottleneck_dim=256
 cfg.ibot.head_n_prototypes=16384
 cfg.ibot.head_hidden_dim=2048
 cfg.ibot.head_bottleneck_dim=256
 cfg.crops.global_crops_size=256
 cfg.crops.local_crops_size=112
 cfg.crops.local_crops_number=8
 cfg.crops.horizontal_flips=False
 cfg.gram.use_loss=False
 cfg.evaluation.eval_period_iterations=max(steps+1,1000000)
 cfg.checkpointing.period=min(1000,max(steps,1))
 cfg.checkpointing.max_to_keep=3
 cfg.checkpointing.keep_every=10000
 cfg.train.saveckp_freq=10000
 cfg.schedules.lr.start=0
 cfg.schedules.lr.peak=2e-5
 cfg.schedules.lr.end=2e-6
 cfg.schedules.lr.warmup_epochs=min(1000,max(1,steps//20))
 cfg.schedules.lr.freeze_last_layer_epochs=min(200,max(1,steps//100))
 cfg.schedules.teacher_temp.start=0.04
 cfg.schedules.teacher_temp.peak=0.07
 cfg.schedules.teacher_temp.end=0.07
 cfg.schedules.teacher_temp.warmup_epochs=min(1000,max(1,steps//20))
 cfg.schedules.momentum.start=0.994
 cfg.schedules.momentum.peak=0.994
 cfg.schedules.momentum.end=1.0
 if arch=='vit_huge2':
  cfg.student.qkv_bias=True
  cfg.student.ffn_layer='swiglu'
  cfg.student.untie_global_and_local_cls_norm=False
 return cfg
