"""Synthetic full student/teacher/optimizer DINOv3 test, never clinical training."""
import argparse,datetime,math,os,socket,time
from pathlib import Path
import torch
import torch.distributed as dist
from runtime import ROOT,save
from configs import config
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--arch',default='vit_7b');p.add_argument('--batch',type=int,default=4);a=p.parse_args()
import dinov3.distributed as distributed
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.data import collate_data_and_cast,MaskingGenerator
from dinov3.checkpointer import save_checkpoint,load_checkpoint

torch.set_num_threads(4);torch.manual_seed(20261015)
distributed.enable(timeout=datetime.timedelta(seconds=600),restrict_print_to_main_process=False)
rank=dist.get_rank();world=dist.get_world_size();cfg=config(a.arch,a.batch,6,a.output)
with torch.device('meta'):model=SSLMetaArch(cfg)
model.prepare_for_distributed_training()
model._apply(lambda t:torch.full_like(t,fill_value=math.nan if t.dtype.is_floating_point else 0,device='cuda'),recurse=True)
model.init_weights();model.train()
optimizer=torch.optim.AdamW(model.student.parameters(),lr=1e-5,betas=(.9,.999),foreach=True)
mask_generator=MaskingGenerator((16,16),max_num_patches=128)
samples=[({'global_crops':[torch.randn(3,256,256) for _ in range(2)],'local_crops':[torch.randn(3,112,112) for _ in range(8)]},0) for _ in range(a.batch)]
data=collate_data_and_cast(samples,(.1,.5),.5,torch.bfloat16,256,mask_generator)
data['global_batch_size']=a.batch*world
step_times=[]
for iteration in range(6):
 optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model.forward_backward(data,teacher_temp=.07,iteration=iteration)
 if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite_synthetic_loss')
 norm=torch.nn.utils.clip_grad_norm_(model.student.parameters(),3.)
 if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite_synthetic_gradient')
 optimizer.step();model.update_ema(.994);torch.cuda.synchronize();step_times.append(time.monotonic()-start)
 save(Path(a.output)/f'rank_{rank:02d}.progress.json',{'state':'capacity_testing','iteration':iteration,'loss_finite':True,'step_seconds':step_times[-1],'peak_memory_GiB':round(torch.cuda.max_memory_allocated()/2**30,2),'clinical_data_accessed':False})
# A small sharded checkpoint verifies the same checkpoint APIs without duplicating 100GB test models.
probe=torch.nn.Linear(4,4,device='cuda');probe_opt=torch.optim.AdamW(probe.parameters())
ckpt=Path(a.output)/'synthetic_checkpoint_test'
save_checkpoint(ckpt_dir=ckpt,iteration=5,model=probe,optimizer=probe_opt)
if load_checkpoint(ckpt,model=probe,optimizer=probe_opt)!=5:raise RuntimeError('checkpoint_roundtrip_failed')
save(Path(a.output)/f'rank_{rank:02d}.json',{'state':'passed','rank':rank,'world_size':world,'host':socket.gethostname(),'architecture':a.arch,'batch_size_per_gpu':a.batch,'peak_memory_GiB':round(torch.cuda.max_memory_allocated()/2**30,2),'step_seconds':step_times,'checkpoint_api_roundtrip':True,'clinical_data_accessed':False,'published_weights_loaded':False,'not_a_trained_encoder':True})
dist.barrier();distributed.disable()
