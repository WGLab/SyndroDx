"""Keep useful GPU children alive if a busy QC slot misses the idle-only switch."""
import fcntl,subprocess,time,traceback
from runtime import ROOT,RUN,SLURM,load,save,scheduler,environment,now,notify
from crop_delegation import ACCEL,PIPELINE
from accelerate_crops import start,step

def main():
 lock=(ACCEL/'guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 replacements={};retry={};merged={};switched=set()
 while True:
  plan=load(ACCEL/'plan.json');status=load(ACCEL/'controller_status.json')
  if not plan.get('claims_sealed'):time.sleep(10);continue
  if status.get('state')!='attention_required':
   if plan.get('state')=='complete' and not replacements:return
   time.sleep(10);continue
  jobs,busy=scheduler();registry=load(ROOT/'independent_qc/launches.json')
  for slot in plan['worker_slots']:
   state=load(ROOT/'independent_qc'/f'slot_{slot:02d}.json');job=registry[str(slot)]['job_id']
   if not load(ACCEL/f'slot_{slot:02d}.switched.json'):
    # The original controller is deliberately retained so its ten useful GPU
    # children and resource locks survive. Replace only this lagging QC process.
    if slot not in switched:
     p=start(step(job,'replace_qc_checkpoint.py',['--slot',str(slot)],True),'checkpoint_switch_'+str(slot))
     if p.wait()!=0:raise RuntimeError('checkpoint_QC_handoff_failed')
     switched.add(slot)
    continue
   if state.get('state')=='finished':continue
   if job in busy:continue
   if slot in replacements and replacements[slot].poll() is None:continue
   retry[slot]=retry.get(slot,0)+1
   if retry[slot]>3:raise RuntimeError('guard_worker_retry_limit')
   current=load(ROOT/'independent_qc/launches.json');current[str(slot)].update(mode='shared_primary_crop_and_independent_QC',launched_at=time.time(),attempts=1);save(ROOT/'independent_qc/launches.json',current)
   replacements[slot]=start(step(job,'dual_crop_qc_worker.py',['--slot',str(slot)]),'guard_worker_'+str(slot))
  original=load(PIPELINE/'handoff_launches.json')
  for owner in range(6):
   st=load(ACCEL/f'owner_{owner:02d}.merged.json')
   if st.get('state')=='complete':continue
   if owner in merged and merged[owner].poll() is None:continue
   f=(ACCEL/f'owner_{owner:02d}.merge.lock').open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:f.close();continue
   f.close();job=original[str(owner)]['allocation_job_id']
   merged[owner]=start(step(job,'delegation_task.py',['--kind','merge','--owner',str(owner),'--expected-host',jobs[job]['node']],True),'guard_merge_'+str(owner))
  if all(load(ACCEL/f'owner_{i:02d}.merged.json').get('state')=='complete' for i in range(6)):
   plan.update(state='complete',completed_at_utc=now());save(ACCEL/'plan.json',plan)
  states=[load(ROOT/'independent_qc'/f'slot_{i:02d}.json').get('state') for i in plan['worker_slots']]
  save(ACCEL/'guard_status.json',{'state':'supervising_shared_workers','shared_workers':sum(s in ['assisting_primary_crop','reviewing'] for s in states),'primary_complete':plan.get('state')=='complete','updated_at_utc':now()})
  if all(s=='finished' for s in states):
   # Give all GPU children time to exit before ending their old CPU supervisor.
   time.sleep(30)
   if any(p.poll() is None for p in replacements.values()) or any(p.poll() is None for p in merged.values()):continue
   old=load(ACCEL/'submission.json').get('job_id')
   if old:
    q=subprocess.run([str(SLURM/'squeue'),'-h','-j',old,'-o','%j'],capture_output=True,text=True,timeout=25)
    if q.stdout.strip()=='ssl-crop-acceleration':subprocess.run([str(SLURM/'scancel'),old],check=True,timeout=25)
   save(ACCEL/'guard_status.json',{'state':'complete','updated_at_utc':now()});return
  time.sleep(10)
if __name__=='__main__':
 try:main()
 except Exception as e:
  with (ACCEL/'guard.private_errors.log').open('a') as f:traceback.print_exc(file=f)
  save(ACCEL/'guard_status.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()})
  while True:time.sleep(30)
