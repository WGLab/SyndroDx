"""Persistent Slurm-owned handoff: crop -> automated QC -> split -> continual SSL."""
import argparse,fcntl,os,subprocess,time,traceback
from pathlib import Path
from runtime import ROOT,RUN,SLURM,load,save,now,scheduler,claim_jobs,environment,notify
QC_COUNT=11

def submit_cpu_stage(kind):
 folder=ROOT/'stage_tasks';folder.mkdir(exist_ok=True)
 status=load(folder/(kind+'.json'));request=load(folder/(kind+'_submission.json'))
 if status.get('state')=='complete':return
 if request.get('job_id'):
  q=subprocess.run([str(SLURM/'squeue'),'-h','-j',str(request['job_id']),'-o','%T'],capture_output=True,text=True,timeout=25)
  if q.returncode==0 and q.stdout.strip():return
  if request.get('attempts',0)>=3:
   notify('SSL '+kind+' stage requires attention','SSL data preparation requires attention','A local validation or release stage could not complete after automatic retries. Other processing and existing GPU reservations are retained. Private diagnostics remain on HPC.');return
 body=folder/(kind+'.sbatch')
 body.write_text('#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true DO_NOT_TRACK=1\nexec '+str(ROOT/'venv/bin/python')+' '+str(ROOT/'stage_task.py')+' --kind '+kind+'\n')
 command=[str(SLURM/'sbatch'),'--parsable','--partition=defq','--job-name=ssl-data-'+kind,'--cpus-per-task=2','--mem=24G','--time=1-00:00:00','--chdir='+str(ROOT),'--output='+str(folder/(kind+'.slurm.private.log')),str(body)]
 result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
 job=result.stdout.strip().split(';')[0]
 if not job.isdigit():raise RuntimeError('cpu_stage_submission_failed')
 save(folder/(kind+'_submission.json'),{'job_id':job,'attempts':request.get('attempts',0)+1,'submitted_at_utc':now()})

def start_qc(jobs,busy):
 stage=ROOT/'independent_qc';stage.mkdir(exist_ok=True);registry=load(stage/'launches.json')
 idle=[v for v in jobs.values() if v['state']=='RUNNING' and v['job_id'] not in busy and not v['name'].startswith('facecrop-') and v['gpus']==1]
 idle.sort(key=lambda v:(v['node'],int(v['job_id'])))
 for slot in range(QC_COUNT):
  state=load(stage/f'slot_{slot:02d}.json');previous=registry.get(str(slot),{})
  if state.get('state')=='finished':continue
  if previous.get('job_id') in busy:continue
  if time.time()-previous.get('launched_at',0)<120:continue
  if previous.get('attempts',0)>=3:
   notify('independent crop QC attention '+str(slot),'Crop QC requires attention','An independent automated crop-review worker needs attention after retries. Other local workers continue; images and detailed diagnostics remain on HPC.');continue
  if not idle:break
  job=idle.pop(0)
  try:locks=claim_jobs([job['job_id']])
  except BlockingIOError:continue
  try:
   command=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--exact','--exclusive','--mpi=none',str(RUN/'crop_pipeline/venv/bin/python'),str(ROOT/'independent_qc.py'),'--slot',str(slot)]
   with (stage/f'slot_{slot:02d}.launch.private.log').open('ab') as f:
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment(),start_new_session=True)
   registry[str(slot)]={'job_id':job['job_id'],'launcher_pid':p.pid,'launched_at':time.time(),'attempts':previous.get('attempts',0)+1}
   busy.add(job['job_id']);save(stage/'launches.json',registry)
  finally:
   for f in locks:f.close()


def tick():
 jobs,busy=scheduler()
 active=[v for v in jobs.values() if v['state']=='RUNNING'];pending=[v for v in jobs.values() if v['state']=='PENDING']
 save(ROOT/'resource_status.json',{'state':'tracked','active_H100s':sum(v['gpus'] for v in active),'pending_H100s':sum(v['gpus'] for v in pending),'allocations':list(jobs.values()),'updated_at_utc':now()})
 if load(ROOT/'dataset_release/identity_plan_status.json').get('state')!='patient_identity_plan_ready':submit_cpu_stage('identity')
 hardware=load(ROOT/'hardware_validation_ready.json')
 if hardware.get('state')!='passed':return 'waiting_for_training_validation'
 start_qc(jobs,busy)
 if load(RUN/'crop_completion.json').get('state')!='crop_complete':return 'cropping_and_independent_QC_in_parallel'
 if any(load(ROOT/'independent_qc'/f'slot_{i:02d}.json').get('state')!='finished' for i in range(QC_COUNT)):return 'independent_automated_QC'
 if load(ROOT/'dataset_release/release.json').get('state')!='ssl_data_ready':
  if load(ROOT/'assets/fairface/status.json').get('state')!='public_reference_ready':return 'waiting_for_public_reference'
  if load(ROOT/'dataset_release/identity_plan_status.json').get('state')!='patient_identity_plan_ready':return 'waiting_for_patient_identity_plan'
  submit_cpu_stage('release');return 'patient_split_and_training_release'
 for marker,event,title,message in [
  ('crop_validation_completion.json','automatic independent crop validation','Crop validation','The independent automated crop-only review and exact native pixel-integrity checks have finished. This is automated QC, not a claim of human adjudication.'),
  ('dataset_split_completion.json','automatic protected patient split','Dataset split','Patient groups and protected pools have been checked, and local training, validation and test manifests are frozen.'),
  ('ssl_data_ready.json','automatic SSL training release','SSL dataset ready','The local training manifests have passed automated release checks. Ordinary reference faces are included without claiming clinical healthy-control status.')]:
  if (RUN/marker).exists():notify(event,title,message)
 training=load(ROOT/'training/state.json');launcher=load(ROOT/'training/launcher.json')
 if training.get('state')=='complete' and (ROOT/'encoder/encoder_export.json').exists():
  save(RUN/'ssl_training_completion.json',{'state':'ssl_training_complete','encoder_path':str(ROOT/'encoder'),'completed_at_utc':now()})
  notify('autonomous SSL pretraining completed','Craniofacial SSL pretraining','The configured continual DINOv3 training run and local encoder export have completed. Checkpoints, manifests, and all private data remain on HPC. Independent evaluation is the next stage.')
  return 'ssl_training_complete'
 if launcher.get('state')=='running' and not any(j in busy for j in launcher.get('allocation_job_ids',[])):
  previous=load(ROOT/'training/controller_launch.json')
  if time.time()-previous.get('launched_at',0)>180:
   launcher['state']='segment_complete' if training.get('state') in ['segment_complete','complete'] else 'attention_required'
   save(ROOT/'training/launcher.json',launcher)
 if launcher.get('state')=='running':
  if training.get('state')=='training':notify('autonomous SSL training started','SSL training started','Continual DINOv3 training has started locally using the available H100 pool. Periodic checkpoints and local logs are enabled; no images or MRI are sent to an external model or tracker.')
  return 'SSL_pretraining'
 launch_info=load(ROOT/'training/controller_launch.json')
 if time.time()-launch_info.get('launched_at',0)<180:return 'waiting_for_training_startup'
 if any(v['job_id'] in busy for v in active):return 'waiting_for_active_H100_processing_to_finish'
 failures=launch_info.get('failures',0)
 if launcher.get('state')=='attention_required':
  failures+=1
  if failures>3:
   notify('autonomous SSL launch attention','SSL pretraining requires attention','Automatic recovery could not resume a local distributed training segment. Existing reservations and data are retained; private diagnostics remain on HPC.');return 'training_requires_attention'
  # Memory failures can recover with a smaller batch while preserving the checkpoint.
  from omegaconf import OmegaConf
  config_path=ROOT/'training/run_config.yaml'
  if config_path.exists() and any(v.get('out_of_memory') for p in (ROOT/'training').glob('rank_*.failure.json') for v in [load(p)]):
   cfg=OmegaConf.load(config_path);cfg.train.batch_size_per_gpu=max(1,cfg.train.batch_size_per_gpu//2);OmegaConf.save(cfg,config_path)
 with (ROOT/'training_launcher.private.log').open('ab') as f:
  p=subprocess.Popen([str(ROOT/'venv/bin/python'),str(ROOT/'launch_training.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment(),start_new_session=True)
 save(ROOT/'training/controller_launch.json',{'pid':p.pid,'launched_at':time.time(),'failures':failures})
 return 'launching_distributed_pretraining'


def main(loop):
 ROOT.mkdir(parents=True,exist_ok=True)
 lock=(ROOT/'controller.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 while True:
  try:
   state=tick();save(ROOT/'controller_status.json',{'state':state,'updated_at_utc':now()})
   if state=='ssl_training_complete':return
  except Exception as e:
   with (ROOT/'controller.private_errors.log').open('a') as f:traceback.print_exc(file=f)
   save(ROOT/'controller_status.json',{'state':'recovering','error_type':type(e).__name__,'updated_at_utc':now()})
  if not loop:return
  time.sleep(45)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--loop',action='store_true');a=p.parse_args();main(a.loop)
