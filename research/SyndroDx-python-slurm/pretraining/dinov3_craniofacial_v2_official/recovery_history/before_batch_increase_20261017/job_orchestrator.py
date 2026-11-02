"""Available-pool launcher with one checkpoint expansion to20; optimization remains upstream official.main()."""
import os,sys,subprocess,time,signal,contextlib,json
from pathlib import Path
from common import ROOT,PYTHON,SLURM,load,save,now,environment,notify
from allocations import active_allocations


def run(phase,batch,pool):
 subprocess.run([str(PYTHON),str(ROOT/'configure_run.py'),'--phase','capacity' if phase=='capacity' else 'main','--batch',str(batch),'--world',str(len(pool))],check=True,env=environment())
 folder=ROOT/'capacity'/f'batch_{batch}' if phase=='capacity' else ROOT/'training'
 save(ROOT/'phase.json',dict(state='capacity_profile' if phase=='capacity' else 'training_initializing',batch_size_per_gpu=batch,world_size=len(pool),updated_at_utc=now()))
 clients=[]
 try:
  for rank,allocation in enumerate(pool):
   env=environment();env.update(CPT_PHASE=phase,CPT_BATCH=str(batch),CPT_RANK=str(rank),CPT_WORLD=str(len(pool)),CPT_MASTER=pool[0]['node'],CPT_PORT=str(35000+int(os.environ['SLURM_JOB_ID'])%20000))
   command=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--kill-on-bad-exit=1','--nodes=1','--ntasks=1','--cpus-per-task=4','--gpus-per-task=1','--gpu-bind=single:1','--mem-per-gpu=68G','--cpu-bind=none','--exact','--exclusive','--mpi=none',str(PYTHON),str(ROOT/'rank_launcher.py'),'--config-file',str(folder/'config.yaml'),'--output-dir',str(folder),'--seed','20261017']
   with (folder/f'rank_{rank:02d}.launcher.private.log').open('ab') as output:
    clients.append(subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True))
  save(ROOT/'active_clients.json',dict(state='running',phase=phase,allocation_job_ids=[a['job_id'] for a in pool],client_pids=[p.pid for p in clients],started_at_utc=now()))
  resize_target=None
  while True:
   codes=[p.poll() for p in clients]
   if any(code not in (None,0) for code in codes):return 1
   if all(code==0 for code in codes):return 0
   if phase=='training' and len(pool)<20:
    available,_=active_allocations()
    if len(available)<20:resize_target=None
    else:
     if resize_target is None:
      last=-1;metric_path=ROOT/'training/training_metrics.json'
      if metric_path.exists():
       with metric_path.open('rb') as f:
        f.seek(max(0,metric_path.stat().st_size-16000));lines=f.read().decode(errors='replace').splitlines()
       for line in reversed(lines):
        try:last=int(json.loads(line)['iteration']);break
        except (ValueError,KeyError):continue
      resize_target=((max(0,last)//100)+1)*100-1
      save(ROOT/'resource_resize.json',dict(state='waiting_for_next_committed_checkpoint',target_checkpoint_iteration=resize_target,current_world_size=len(pool),requested_world_size=20,requested_at_utc=now()))
     directory=ROOT/'training/ckpt'/str(resize_target)
     shards=list(directory.glob('*.distcp')) if directory.is_dir() else []
     if (directory/'.metadata').is_file() and (directory/'.metadata').stat().st_size>0 and len(shards)>=len(pool) and all(p.stat().st_size>0 for p in shards):
      save(ROOT/'resource_resize.json',dict(state='committed_checkpoint_handoff',committed_checkpoint_iteration=resize_target,current_world_size=len(pool),requested_world_size=20,parent_allocations_cancelled=False,updated_at_utc=now()))
      return 75
   time.sleep(3)
 finally:
  for p in clients:
   if p.poll() is None:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
  for p in clients:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
  save(ROOT/'active_clients.json',dict(state='exited',phase=phase,allocation_job_ids=[a['job_id'] for a in pool],updated_at_utc=now()))



def reusable_capacity(root,world):
 certificate=load(root/'capacity/selection.json')
 batch=certificate.get('batch_size_per_gpu')
 committed=[p for p in (root/'training/ckpt').glob('*') if p.is_dir() and p.name.isdigit() and (p/'.metadata').is_file()]
 if not committed or certificate.get('state')!='passed' or certificate.get('world_size')!=world or batch not in [64,32,16]:return None
 if not certificate.get('full_official_heads') or not certificate.get('all_ranks_checked'):return None
 receipts=[load(root/'capacity'/f'batch_{batch}'/f'rank_{i}.json') for i in range(world)]
 if not all(x.get('state')=='passed' and x.get('peak_reserved_GiB',999)<74 for x in receipts):return None
 return batch


def main():
 os.umask(0o077)
 if load(ROOT/'initialization/ready.json').get('state')!='ready':raise RuntimeError('CPU_preparation_not_ready')
 deadline=time.monotonic()+2*86400
 while True:
  pool,_=active_allocations()
  minimum=int(load(ROOT/'start_policy.json').get('minimum_starting_GPUs',20))
  if len(pool)>=minimum:break
  save(ROOT/'phase.json',dict(state='waiting_for_starting_H100_allocations',running_allocations=len(pool),minimum_starting_allocations=minimum,target_allocations=20,updated_at_utc=now()))
  if time.monotonic()>deadline:raise RuntimeError('full20_GPU_pool_wait_timeout')
  time.sleep(30)
 save(ROOT/'selected_pool.json',dict(state='selected_initial_pool',world_size=len(pool),allocations=pool,selected_at_utc=now()))
 chosen=reusable_capacity(ROOT,len(pool))
 if chosen is None:
  for batch in [64,32,16]:
   code=run('capacity',batch,pool)
   receipts=[load(ROOT/'capacity'/f'batch_{batch}'/f'rank_{i}.json') for i in range(len(pool))]
   if code==0 and all(x.get('state')=='passed' and x.get('peak_reserved_GiB',999)<74 for x in receipts):chosen=batch;break
   save(ROOT/'capacity'/f'batch_{batch}'/'rejected.json',dict(state='not_selected',launcher_exit_code=code,all_ranks_passed=all(x.get('state')=='passed' for x in receipts),updated_at_utc=now()))
   if code and not any(x.get('out_of_memory') for x in receipts):raise RuntimeError('synthetic_preflight_failed_not_a_batch_capacity_issue')
 if chosen is None:raise RuntimeError('no_safe_full_official_model_batch')
 save(ROOT/'capacity/selection.json',dict(state='passed',batch_size_per_gpu=chosen,world_size=len(pool),full_official_heads=True,all_ranks_checked=True,clinical_data_accessed=False,selected_at_utc=now()))
 notify('training launch '+os.environ.get('SLURM_JOB_ID','unknown'),'Official-code craniofacial CPT launch','The allocated GPU pool has a passed synthetic capacity check. The unchanged official trainer is starting or resuming from its latest committed model and optimizer checkpoint. Fixed patient-isolated validation remains scheduled every100steps. All clinical data and detailed metrics remain on HPC.')
 while True:
  code=run('training',chosen,pool)
  if code==75:
   pool,_=active_allocations()
   if len(pool)!=20:raise RuntimeError('GPU_pool_changed_during_checkpoint_handoff')
   save(ROOT/'selected_pool.json',dict(state='expanded_at_committed_checkpoint',world_size=len(pool),allocations=pool,updated_at_utc=now()))
   notify('checkpoint pool expansion','Official-code CPT checkpoint resume','The20-GPU pool is available. The project training steps have stopped at a committed checkpoint and are resuming with the official model and optimizer loader. Parent GPU allocations and the original iteration schedule are preserved. Detailed results remain on HPC.')
   continue
  if code:raise RuntimeError('official_training_process_failed')
  break
 save(ROOT/'training/completion.json',dict(state='training_complete',world_size=len(pool),completed_at_utc=now(),final_validation=load(ROOT/'validation/latest.json')))
 save(ROOT/'phase.json',dict(state='training_complete',world_size=len(pool),allocations_retained=True,updated_at_utc=now()))
 notify('training complete','Official-code craniofacial CPT training completed','The configured official-code continual-training stage and scheduled validation have completed. This does not establish clinical usefulness or superiority. CPU encoder export follows; private results remain on HPC.')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback;traceback.print_exc()
  save(ROOT/'failure.json',dict(state='failed',error_type=type(exc).__name__,updated_at_utc=now()))
  notify('training failure '+os.environ.get('SLURM_JOB_ID','unknown'),'Official-code CPT requires attention','The official-code training or its synthetic resource check did not complete. Checkpoints and the original encoder are preserved. Private diagnostic logs remain on HPC; no automatic recipe changes were made.')
  raise SystemExit(1)
