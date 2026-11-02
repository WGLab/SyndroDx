"""Explicit resource overrides; keep the upstream iteration schedule on resume."""
import argparse,math,sys,shutil
from common import ROOT,V1,load,save,now
from omegaconf import OmegaConf
from prepare import configuration

def resumed_configuration(cfg,batch,world):
 cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
 cfg.train.batch_size_per_gpu=batch
 scale=4*math.sqrt(batch*world/1024)
 cfg.schedules.lr.peak=2e-6/scale;cfg.schedules.lr.end=2e-7/scale
 return cfg


def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['capacity','main'],required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--world',type=int,required=True);a=p.parse_args()
 if a.phase=='capacity':
  cfg=configuration(a.batch,4,world=a.world);cfg.schedules.lr.warmup_epochs=1;cfg.schedules.lr.freeze_last_layer_epochs=1;cfg.schedules.teacher_temp.warmup_epochs=1
  cfg.evaluation.eval_period_iterations=0;cfg.checkpointing.period=4
  folder=ROOT/'capacity'/f'batch_{a.batch}';folder.mkdir(parents=True,exist_ok=True);cfg.train.output_dir=str(folder)
 else:
  folder=ROOT/'training';folder.mkdir(exist_ok=True)
  if (folder/'config.yaml').exists():
   cfg=OmegaConf.load(folder/'config.yaml');previous=load(ROOT/'run_plan.json')
   backup=folder/('config.world_'+str(previous.get('world_size','unknown'))+'.batch_'+str(previous.get('batch_size_per_gpu','unknown'))+'.yaml')
   if not backup.exists():shutil.copyfile(folder/'config.yaml',backup)
   cfg=resumed_configuration(cfg,a.batch,a.world)
   history=previous.get('world_history',[])
   committed=[int(p.name) for p in (folder/'ckpt').glob('*') if p.is_dir() and p.name.isdigit() and (p/'.metadata').is_file()]
   resumed_from=max(committed) if committed else load(ROOT/'batch_resize_request.json').get('target_checkpoint_iteration')
   if not history or history[-1]['world_size']!=a.world or history[-1].get('batch_size_per_gpu')!=a.batch:history.append(dict(world_size=a.world,batch_size_per_gpu=a.batch,resume_checkpoint=resumed_from,at_utc=now()))
   save(ROOT/'run_plan.json',dict(previous,world_size=a.world,batch_size_per_gpu=a.batch,global_batch_size=a.batch*a.world,world_history=history,resource_resize_policy='official DCP model/optimizer resume; original optimizer-step schedule retained; realized image exposure changes with global batch and must be reported',updated_at_utc=now()))
  else:
   old=load(V1/'training/run_plan.json');target=old['total_iterations']*old['initial_world_size']*old['batch_size_per_gpu']
   total=math.ceil(target/(a.world*a.batch)/100)*100;cfg=configuration(a.batch,total,world=a.world)
   save(ROOT/'run_plan.json',dict(state='configured',initial_world_size=a.world,world_size=a.world,batch_size_per_gpu=a.batch,global_batch_size=a.world*a.batch,total_steps=total,checkpoint_and_validation_period=100,original_target_sample_budget=target,target_sample_budget_source='v1 originally declared exposure budget, rounded up to next100step boundary for initial pool',domain_effective_peak_lr=2e-6,domain_effective_end_lr=2e-7,official_scheduler_scaling_retained=True,heads='unchanged official ViT7B reference dimensions',extra_domain_losses=False,world_history=[dict(world_size=a.world,batch_size_per_gpu=a.batch,at_utc=now())],created_at_utc=now()))
 OmegaConf.save(cfg,folder/'config.yaml')

if __name__=='__main__':main()
