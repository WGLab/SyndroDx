"""Recover only terminal QC failures caused by a transient source snapshot read."""
import fcntl,json,subprocess,time
from pathlib import Path
from runtime import ROOT,RUN,SLURM,load,save,scheduler,claim_jobs,environment

def main():
 lock=(ROOT/'independent_qc/transient_recovery.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 jobs,busy=scheduler();registry=load(ROOT/'independent_qc/launches.json')
 idle=[v for v in jobs.values() if v['state']=='RUNNING' and v['job_id'] not in busy and v['gpus']==1 and not v['name'].startswith('facecrop-')]
 for slot in range(11):
  folder=ROOT/'independent_qc';state=load(folder/f'slot_{slot:02d}.json');previous=registry.get(str(slot),{})
  if state.get('state')!='attention_required' or time.time()-previous.get('launched_at',0)<180:continue
  error=folder/f'slot_{slot:02d}.private_errors.log'
  try:
   lines=error.read_text(errors='replace').splitlines()
   endings=[v for v in lines if v.startswith(('RuntimeError:','ValueError:','OSError:','sqlite3.'))]
   if not endings or endings[-1]!='RuntimeError: source_snapshot_unavailable':continue
  except OSError:continue
  # A live worker always owns this slot lock; never interrupt or duplicate it.
  owner=(folder/f'slot_{slot:02d}.lock').open('a')
  try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:owner.close();continue
  if not idle:owner.close();return
  job=idle.pop(0)
  try:locks=claim_jobs([job['job_id']])
  except BlockingIOError:owner.close();continue
  try:
   registry=load(folder/'launches.json');registry[str(slot)]={'job_id':job['job_id'],'attempts':1,'launched_at':time.time(),'recovery_reason':'transient_source_snapshot_retry_in_worker','prior_attempts':previous.get('attempts',0)}
   save(folder/'launches.json',registry);owner.close()
   command=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--exact','--exclusive','--mpi=none',str(RUN/'crop_pipeline/venv/bin/python'),str(ROOT/'independent_qc.py'),'--slot',str(slot)]
   with (folder/f'slot_{slot:02d}.launch.private.log').open('ab') as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment(),start_new_session=True)
   registry[str(slot)]['launcher_pid']=p.pid;save(folder/'launches.json',registry)
  finally:
   owner.close()
   for f in locks:f.close()

if __name__=='__main__':main()
