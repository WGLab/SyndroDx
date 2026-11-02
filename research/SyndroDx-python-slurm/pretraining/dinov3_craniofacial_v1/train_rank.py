"""Offline continual DINOv3 with FSDP2 and resumable, finite training segments."""
import argparse,datetime,json,math,os,time,traceback
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from omegaconf import OmegaConf
from runtime import ROOT,RUN,save,load,now
from weights import OfficialBackbone
from batch_policy import schedule_index,sampler_advance
from dataset import CraniofacialAugmentation,make_dataset
from dinov3.train.ssl_meta_arch import SSLMetaArch
import dinov3.train.train as upstream
import dinov3.distributed as distributed
from dinov3.checkpointer import find_latest_checkpoint,load_checkpoint,save_checkpoint,keep_last_n_checkpoints

class ContinualMetaArch(SSLMetaArch):
 def __init__(self,cfg):
  super().__init__(cfg)
  # The upstream training factory omits pos_embed_rope_dtype. Match the
  # published checkpoint's FP32 RoPE instead of the constructor's BF16 default.
  for bundle in [self.student,self.teacher]:
   rope=bundle.backbone.rope_embed;rope.dtype=torch.float32;rope.periods=rope.periods.to(torch.float32)
 def build_data_augmentation_dino(self,cfg):return CraniofacialAugmentation(cfg)
 def init_weights(self):
  super().init_weights()
  path=ROOT/'assets/dinov3-vit7b16' if self.cfg.student.arch=='vit_7b' else ROOT/'assets/dinov3-vith16plus'
  checkpoint=OfficialBackbone(path);checkpoint.load(self.student.backbone)
  self.model_ema.load_state_dict(self.student.state_dict())


def run(args):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(20261015)
 distributed.enable(timeout=datetime.timedelta(seconds=600),restrict_print_to_main_process=False)
 rank=dist.get_rank();world=dist.get_world_size();cfg=OmegaConf.load(args.config)
 upstream.make_dataset=make_dataset
 output=Path(cfg.train.output_dir);output.mkdir(parents=True,exist_ok=True)
 if rank==0:
  OmegaConf.save(cfg,output/'resolved_config.yaml')
  save(output/'state.json',{'state':'initializing','world_size':world,'architecture':cfg.student.arch,'updated_at_utc':now()})
 with torch.device('meta'):model=ContinualMetaArch(cfg)
 model.prepare_for_distributed_training()
 model._apply(lambda t:torch.full_like(t,fill_value=math.nan if t.dtype.is_floating_point else 0,device='cuda'),recurse=True)
 model.init_weights();model.train()
 optimizer=upstream.build_optimizer(cfg,model.get_params_groups())
 ckpt=output/'ckpt';ckpt.mkdir(exist_ok=True);start=0
 latest=find_latest_checkpoint(ckpt)
 if latest:start=load_checkpoint(latest,model=model,optimizer=optimizer,strict_loading=True)+1
 schedules=upstream.build_schedulers(cfg)
 base_total=int(cfg.optim.epochs*cfg.train.OFFICIAL_EPOCH_LENGTH)
 policy=load(output/'execution_policy.json')
 if policy:
  if policy.get('state')!='active' or policy.get('global_batch_size')!=cfg.train.batch_size_per_gpu*world or start<policy['anchor_iteration']:raise RuntimeError('execution_policy_mismatch')
  original_loader=upstream.make_data_loader
  def exposure_loader(**kwargs):
   kwargs['sampler_advance']=sampler_advance(policy,start,world)
   return original_loader(**kwargs)
  upstream.make_data_loader=exposure_loader
 try:loader=upstream.build_multi_resolution_data_loader_from_cfg(cfg=cfg,model=model,start_iter=start)
 finally:
  if policy:upstream.make_data_loader=original_loader
 iterator=iter(loader);total=int(policy['total_iterations']) if policy else base_total
 end=min(total,start+args.segment_steps);batch=cfg.train.batch_size_per_gpu*world
 if rank==0:save(output/'state.json',{'state':'training','world_size':world,'start_iteration':start,'segment_end':end,'total_iterations':total,'updated_at_utc':now()})
 last_saved=start-1
 for iteration in range(start,end):
  tick=time.monotonic();data=next(iterator);data['global_batch_size']=batch
  schedule_iteration=schedule_index(policy,iteration,base_total)
  lr,wd,momentum,temp,last_layer=[float(schedule[schedule_iteration]) for schedule in schedules]
  upstream.apply_optim_scheduler(optimizer,lr,wd,last_layer)
  optimizer.zero_grad(set_to_none=True)
  loss,metrics=model.forward_backward(data,teacher_temp=temp,iteration=iteration)
  valid=torch.isfinite(loss.detach()).to(torch.int32);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
  if valid.item()!=1:raise RuntimeError('nonfinite_training_loss')
  for part in model.student.values():
   norm=torch.nn.utils.clip_grad_norm_(part.parameters(),cfg.optim.clip_grad)
   norm=norm.full_tensor() if isinstance(norm,DTensor) else norm
   if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite_training_gradient')
  optimizer.step();model.update_ema(momentum)
  mean_loss=loss.detach().clone();dist.all_reduce(mean_loss);mean_loss/=world
  if rank==0 and (iteration%10==0 or iteration==end-1):
   value={'iteration':iteration,'schedule_iteration':schedule_iteration,'total_loss':float(mean_loss),'lr':lr,'world_size':world,'global_batch_size':batch,'step_seconds':time.monotonic()-tick,'peak_memory_GiB':torch.cuda.max_memory_allocated()/2**30,'updated_at_utc':now()}
   with (output/'metrics.private.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
   save(output/'state.json',dict(value,state='training',segment_end=end,total_iterations=total))
  if (start==0 and iteration==9) or (policy and iteration==start+9) or (iteration+1)%cfg.checkpointing.period==0 or iteration==end-1:
   save_checkpoint(ckpt/str(iteration),iteration=iteration,model=model,optimizer=optimizer,overwrite=False,process_group=distributed.get_process_subgroup())
   last_saved=iteration
   if rank==0:keep_last_n_checkpoints(ckpt,cfg.checkpointing.max_to_keep)
 if end>=total:
  export=output/'synthetic_encoder' if 'synthetic' in str(cfg.student.arch) else ROOT/'encoder'
  export.mkdir(parents=True,exist_ok=True)
  state=model.teacher.backbone.state_dict()
  temporary=export/'craniofacial_encoder.safetensors.tmp'
  writer=None
  if rank==0:
   from streaming_encoder import StreamingEncoderWriter
   writer=StreamingEncoderWriter(temporary,state)
  try:
   for name,tensor in state.items():
    value=tensor.full_tensor() if isinstance(tensor,DTensor) else tensor
    if rank==0:writer.write(name,value)
    del value
   if rank==0:writer.close()
  except BaseException:
   if writer is not None:writer.abort()
   raise
  if rank==0:
   temporary.replace(export/'craniofacial_encoder.safetensors')
   OmegaConf.save(cfg,export/'training_config.yaml')
   save(export/'preprocessing.json',{'mode':'RGB','longest_edge':cfg.crops.global_crops_size,'padding':'outside_original_rectangle_only','rgb_mean':list(cfg.crops.rgb_mean),'rgb_std':list(cfg.crops.rgb_std),'no_shape_warp':True,'background_removal':False,'eligible_views':['frontal','oblique','profile']})
   save(export/'encoder_export.json',{'state':'encoder_export_complete','architecture':cfg.student.arch,'teacher_EMA_exported':True,'preprocessing':'aspect_preserving_full_native_crop','source_config':str(output/'resolved_config.yaml'),'private_checkpoint_remains_local':True,'completed_at_utc':now()})
 if rank==0:
  save(output/'state.json',{'state':'complete' if end>=total else 'segment_complete','last_checkpoint_iteration':last_saved,'next_iteration':end,'total_iterations':total,'world_size':world,'architecture':cfg.student.arch,'updated_at_utc':now()})
 dist.barrier();distributed.disable()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--segment-steps',type=int,default=1000);args=p.parse_args()
 try:run(args)
 except Exception as e:
  out=ROOT/'training'
  try:
   out=Path(OmegaConf.load(args.config).train.output_dir);out.mkdir(parents=True,exist_ok=True)
   with (out/('rank_'+os.environ.get('RANK','0')+'.private_errors.log')).open('a') as f:traceback.print_exc(file=f)
   save(out/('rank_'+os.environ.get('RANK','0')+'.failure.json'),{'state':'rank_failed','error_type':type(e).__name__,'out_of_memory':isinstance(e,torch.OutOfMemoryError),'updated_at_utc':now()})
  finally:raise SystemExit(1)
