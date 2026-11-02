"""Use every idle active H100 allocation, resuming at a safe checkpoint boundary."""
import contextlib,fcntl,hashlib,math,os,secrets,signal,subprocess,time
from pathlib import Path
from runtime import ROOT,RUN,SLURM,load,save,digest,now,scheduler,claim_jobs,environment,notify

def main():
 release=load(ROOT/'dataset_release/release.json')
 if release.get('state')!='ssl_data_ready':raise RuntimeError('released_dataset_required')
 if load(ROOT/'hardware_validation_ready.json').get('state')!='passed':raise RuntimeError('hardware_and_training_smoke_validation_required')
 for m in release['manifests'].values():
  if digest(m['path'])!=m['sha256']:raise RuntimeError('released_manifest_changed')
 jobs,busy=scheduler();selected=[v for v in jobs.values() if v['state']=='RUNNING']
 if not selected or any(v['job_id'] in busy for v in selected):raise RuntimeError('all_active_H100_allocations_must_be_idle')
 selected.sort(key=lambda v:(v['node'],int(v['job_id'])))
 locks=claim_jobs([v['job_id'] for v in selected]);processes=[]
 output=ROOT/'training';output.mkdir(exist_ok=True)
 try:
  from configs import config
  from omegaconf import OmegaConf
  world=sum(v['gpus'] for v in selected)
  # Recheck the exact current pool, including newly freed crop nodes, through
  # the identical rank-entry command used by training. No images are opened.
  pool_key=hashlib.sha256(','.join(v['job_id'] for v in selected).encode()).hexdigest()[:16]
  preflight=ROOT/'hardware_probes/full_training_pool'/pool_key
  if load(preflight/'completion.json').get('state')!='passed':
   preflight.mkdir(parents=True,exist_ok=True)
   master=selected[0]['node'];port=35000+secrets.randbelow(20000);offset=0
   for job in selected:
    command=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--nodes=1','--ntasks='+str(job['gpus']),'--cpus-per-task=4','--gpus-per-task=1','--gpu-bind=single:1','--mem-per-gpu=48G','--exact','--exclusive','--mpi=none',str(ROOT/'venv/bin/python'),str(ROOT/'rank_entry.py'),'--offset',str(offset),'--world',str(world),'--master',master,'--port',str(port),'--run-id','pool-'+pool_key,'--',str(ROOT/'pool_rank_probe.py'),'--output',str(preflight)]
    with (preflight/('allocation_'+job['job_id']+'.private.log')).open('ab') as f:processes.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment(),start_new_session=True))
    offset+=job['gpus']
   deadline=time.monotonic()+300
   while time.monotonic()<deadline:
    codes=[p.poll() for p in processes]
    if any(c not in (None,0) for c in codes):raise RuntimeError('current_full_H100_pool_communication_failed')
    if all(c==0 for c in codes):break
    time.sleep(2)
   else:raise RuntimeError('current_full_H100_pool_preflight_timeout')
   if any(load(preflight/f'rank_{i:02d}.json').get('state')!='passed' for i in range(world)):raise RuntimeError('current_full_H100_pool_receipts_incomplete')
   save(preflight/'completion.json',{'state':'passed','world_size':world,'job_ids':[v['job_id'] for v in selected],'clinical_data_accessed':False,'completed_at_utc':now()})
   processes=[]
  cfg_path=output/'run_config.yaml'
  if not cfg_path.exists():
   capacity=[]
   for p in (ROOT/'hardware_probes').glob('capacity_*/completion.json'):
    value=load(p)
    if value.get('state')=='passed' and value.get('arch')=='vit_7b' and max(r.get('peak_memory_GiB',999) for r in value.get('rank_receipts',[]))<64:capacity.append(value['batch_size_per_gpu'])
   if not capacity:raise RuntimeError('successful_ViT7B_capacity_profile_required')
   batch=max(capacity)
   steps=max(2000,math.ceil(release['clinical_train_images']*20/(.8*batch*world)))
   cfg=config(batch=batch,steps=steps,output=output);cfg.checkpointing.period=250
   OmegaConf.save(cfg,cfg_path)
   save(output/'run_plan.json',{'initial_world_size':world,'batch_size_per_gpu':batch,'total_iterations':steps,'target_clinical_image_equivalent_exposures':20,'patient_and_view_balanced':True,'source_sampling_weights':release['sampling_weights'],'healthy_diagnosis_assertion':False,'created_at_utc':now()})
  from automatic_batch_tuning import prepare
  prepare(selected,cfg_path)
  master=selected[0]['node'];port=35000+secrets.randbelow(20000);run_id='craniofacial-'+secrets.token_hex(6);offset=0
  for job in selected:
   command=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--nodes=1','--ntasks='+str(job['gpus']),'--cpus-per-task=4','--gpus-per-task=1','--gpu-bind=single:1','--mem-per-gpu=48G','--exact','--exclusive','--mpi=none',str(ROOT/'venv/bin/python'),str(ROOT/'rank_entry.py'),'--offset',str(offset),'--world',str(world),'--master',master,'--port',str(port),'--run-id',run_id,'--',str(ROOT/'train_rank.py'),'--config',str(cfg_path),'--segment-steps','1000']
   with (output/('allocation_'+job['job_id']+'.private.log')).open('ab') as f:
    processes.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment(),start_new_session=True))
   offset+=job['gpus']
  save(output/'launcher.json',{'state':'running','world_size':world,'allocation_job_ids':[v['job_id'] for v in selected],'launcher_pids':[p.pid for p in processes],'started_at_utc':now()})
  while True:
   codes=[p.poll() for p in processes]
   if any(c not in (None,0) for c in codes):raise RuntimeError('distributed_training_rank_failed')
   if all(c==0 for c in codes):
    state=load(output/'state.json').get('state')
    if state not in ['segment_complete','complete']:raise RuntimeError('training_segment_receipt_missing')
    save(output/'launcher.json',{'state':state,'world_size':world,'completed_at_utc':now()})
    return
   time.sleep(10)
 except Exception as e:
  save(output/'launcher.json',{'state':'attention_required','error_type':type(e).__name__,'reason':str(e),'updated_at_utc':now()})
  raise
 finally:
  # Tear down only our failed collective group. Allocation parents remain alive.
  for p in processes:
   if p.poll() is None:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
  for p in processes:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
  for f in locks:f.close()

if __name__=='__main__':main()
