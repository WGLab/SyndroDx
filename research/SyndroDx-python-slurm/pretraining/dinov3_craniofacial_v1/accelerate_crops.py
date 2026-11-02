"""Persistent, resumable acceleration using idle reserved QC GPUs only."""
import fcntl,json,os,subprocess,time,traceback,uuid
from runtime import ROOT,RUN,SLURM,load,save,now,scheduler,claim_jobs,environment,notify
from crop_delegation import ACCEL,PIPELINE,distribute
PYTHON=PIPELINE/'venv/bin/python'

def step(job,script,args,cpu=False):
 cmd=[str(SLURM/'srun'),'--jobid='+str(job),'--nodes=1','--ntasks=1','--cpus-per-task=1' if cpu else '--cpus-per-task=4','--mpi=none','--exact']
 cmd+=['--gres=none','--overlap'] if cpu else ['--gres=gpu:1','--exclusive']
 return cmd+[str(PYTHON),str(ROOT/script)]+args

def start(cmd,name):
 with (ACCEL/(name+'.private.log')).open('ab') as f:
  return subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=environment())

def checked(cmd,name):
 p=start(cmd,name)
 if p.wait()!=0:raise RuntimeError(name+'_failed')

def main():
 ACCEL.mkdir(exist_ok=True)
 lock=(ACCEL/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=load(ACCEL/'plan.json')
 if plan.get('state')=='complete':return
 jobs,busy=scheduler();registry=load(ROOT/'independent_qc/launches.json')
 if not plan:
  slots=[i for i in range(11) if registry.get(str(i),{}).get('job_id') in jobs and load(ROOT/'independent_qc'/f'slot_{i:02d}.json').get('state')!='finished']
  if not slots:raise RuntimeError('no_live_QC_allocations')
  plan={'state':'claiming','delegation_id':uuid.uuid4().hex,'worker_slots':slots,'created_at_utc':now()};save(ACCEL/'plan.json',plan)
 slots=plan['worker_slots']
 # Resource locks outlive switched workers and prevent the older coordinator
 # assigning a second worker to the same allocation during the brief switch.
 locks=claim_jobs([registry[str(i)]['job_id'] for i in slots])
 original=load(PIPELINE/'handoff_launches.json')
 if not plan.get('claims_sealed'):
  procs=[]
  for owner in range(6):
   job=original[str(owner)]['allocation_job_id'];node=jobs[job]['node']
   procs.append((owner,start(step(job,'delegation_task.py',['--kind','claim','--owner',str(owner),'--expected-host',node],True),'claim_'+str(owner))))
  for owner,p in procs:
   if p.wait()!=0:raise RuntimeError('original_owner_claim_failed_'+str(owner))
  distribute(slots);plan=load(ACCEL/'plan.json')
 mergers={};workers={};attempts={i:0 for i in slots};switchers={}
 for owner in range(6):
  job=original[str(owner)]['allocation_job_id'];node=jobs[job]['node']
  mergers[owner]=start(step(job,'delegation_task.py',['--kind','merge','--owner',str(owner),'--expected-host',node],True),'merge_'+str(owner))
 for slot in slots:
  state=load(ROOT/'independent_qc'/f'slot_{slot:02d}.json')
  job=registry[str(slot)]['job_id']
  if load(ACCEL/f'slot_{slot:02d}.switched.json').get('state')=='idle_QC_process_replaced':
   # Restart is safe only after the prior child step has disappeared.
   switchers[slot]=None
  else:switchers[slot]=start(step(job,'switch_idle_qc.py',['--slot',str(slot)],True),'switch_'+str(slot))
 complete=False
 while True:
  jobs,busy=scheduler()
  for slot,switch in list(switchers.items()):
   if switch is not None and switch.poll() is None:continue
   if switch is not None and switch.returncode!=0:
    checked(step(registry[str(slot)]['job_id'],'replace_qc_checkpoint.py',['--slot',str(slot)],True),'checkpoint_switch_'+str(slot))
    switchers[slot]=None
    continue
   job=registry[str(slot)]['job_id']
   # Do not overlap a replacement with its terminating predecessor.
   if job in busy:continue
   current=load(ROOT/'independent_qc/launches.json')
   current[str(slot)].update(mode='shared_primary_crop_and_independent_QC',launched_at=time.time(),attempts=1)
   save(ROOT/'independent_qc/launches.json',current)
   workers[slot]=start(step(job,'dual_crop_qc_worker.py',['--slot',str(slot)]),'worker_'+str(slot));attempts[slot]+=1
   del switchers[slot]
  for owner,p in list(mergers.items()):
   if p.poll() is not None and load(ACCEL/f'owner_{owner:02d}.merged.json').get('state')!='complete':
    job=original[str(owner)]['allocation_job_id'];node=jobs[job]['node']
    mergers[owner]=start(step(job,'delegation_task.py',['--kind','merge','--owner',str(owner),'--expected-host',node],True),'merge_'+str(owner))
  for slot,p in list(workers.items()):
   if p.poll() is None:continue
   if load(ROOT/'independent_qc'/f'slot_{slot:02d}.json').get('state')=='finished':continue
   if attempts[slot]>=3:raise RuntimeError('combined_worker_retry_limit_'+str(slot))
   job=registry[str(slot)]['job_id']
   if job not in busy:
    workers[slot]=start(step(job,'dual_crop_qc_worker.py',['--slot',str(slot)]),'worker_'+str(slot));attempts[slot]+=1
  if not complete and all(load(ACCEL/f'owner_{i:02d}.merged.json').get('state')=='complete' for i in range(6)):
   plan=load(ACCEL/'plan.json');plan.update(state='complete',completed_at_utc=now());save(ACCEL/'plan.json',plan);complete=True
   notify('idle GPU crop acceleration complete','Additional crop processing completed','The additional reserved GPUs have completed their assigned primary crop work. Local independent crop review continues; all clinical data remain on HPC.')
  live=sum(p.poll() is None for p in workers.values());waiting=len(switchers)
  save(ACCEL/'controller_status.json',{'state':'running','combined_workers_live':live,'idle_workers_switching':waiting,'primary_delegation_complete':complete,'updated_at_utc':now()})
  if not switchers and all(load(ROOT/'independent_qc'/f'slot_{i:02d}.json').get('state')=='finished' for i in slots) and all(p.poll() is not None for p in mergers.values()):return
  time.sleep(10)

if __name__=='__main__':
 try:main()
 except BlockingIOError:raise SystemExit(0)
 except Exception as e:
  with (ACCEL/'controller.private_errors.log').open('a') as f:traceback.print_exc(file=f)
  save(ACCEL/'controller_status.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()})
  # Stay alive: exiting a CPU Slurm job would terminate its useful GPU children.
  while True:time.sleep(30)
