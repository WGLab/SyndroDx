"""Distribute the remaining review straggler across reserved idle H100s."""
import fcntl,subprocess,time,traceback
from runtime import ROOT,RUN,load,save,now,scheduler,notify
from accelerate_crops import start,step
TAIL=ROOT/'qc_tail'
def gpu_step(job,script,args):
 cmd=step(job,script,args);cmd.remove('--exclusive');cmd.insert(1,'--overlap');return cmd

def main():
 lock=(TAIL/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 existing=load(TAIL/'plan.json')
 if existing and existing.get('state') not in ['running','complete']:raise RuntimeError('partial_tail_plan_requires_audit')
 jobs,busy=scheduler();qc=load(ROOT/'independent_qc/launches.json');original=load(RUN/'crop_pipeline/handoff_launches.json')
 base=list(range(11));extras=list(range(11,17));recipients=base+extras
 # All six initial primary allocations have finished both their original and
 # additional crop partitions; retain the allocations and use them for review.
 for i in range(6):
  if load(RUN/'crop_pipeline/workers'/f'slot_{i:02d}.json').get('state')!='finished_provisional':raise RuntimeError('released_primary_GPU_required')
 if not existing:save(TAIL/'plan.json',{'state':'preparing','base_slots':base,'extra_slots':extras,'recipients':recipients,'created_at_utc':now()})
 controls={};manual={};aux={};attempts={};auto={e['source_slot'] for e in load(ROOT/'crop_acceleration/rebalancing.json').get('extras',{}).values() if e.get('state')=='running'}
 def checkpoint(slot):
  entry=qc[str(slot)];job=entry['job_id'];state=load(ROOT/'independent_qc'/f'slot_{slot:02d}.json')
  state.update(state='preparing_review_redistribution',updated_at=time.time());save(ROOT/'independent_qc'/f'slot_{slot:02d}.json',state)
  return start(step(job,'checkpoint_qc_tail.py',['--slot',str(slot),'--expected-host',jobs[job]['node']],True),'qc_tail_checkpoint_'+str(slot))
 if not existing:
  controls[1]=checkpoint(1)
  for _ in range(600):
   if load(TAIL/'base_01.checkpoint.json').get('state')=='checkpoint_saved':break
   if controls[1].poll() is not None and controls[1].returncode!=0:raise RuntimeError('review_tail_assignment_failed')
   time.sleep(.2)
  else:raise RuntimeError('review_tail_assignment_timed_out')
 for slot in base:
  if slot!=1 and not load(TAIL/f'base_{slot:02d}.checkpoint.json'):controls[slot]=checkpoint(slot)
 for slot in extras:
  job=original[str(slot-11)]['allocation_job_id'];aux[slot]=start(gpu_step(job,'extra_qc_tail_worker.py',['--slot',str(slot)]),'qc_tail_extra_'+str(slot))
 source_job=qc['1']['job_id'];source_host=jobs[source_job]['node']
 bridge=start(step(source_job,'bridge_qc_tail.py',['--expected-host',source_host],True),'qc_tail_bridge')
 completed=False
 while True:
  for slot,p in list(controls.items()):
   receipt=load(TAIL/f'base_{slot:02d}.checkpoint.json')
   if receipt.get('state')=='checkpoint_saved' and slot not in auto and slot not in manual:
    job=qc[str(slot)]['job_id'];manual[slot]=start(gpu_step(job,'dual_crop_qc_worker.py',['--slot',str(slot)]),'qc_tail_base_'+str(slot))
   if p.poll() is not None and p.returncode!=0:raise RuntimeError('review_checkpoint_restart_failed_'+str(slot))
  for slot,p in list(manual.items()):
   if p.poll() is None or load(ROOT/'independent_qc'/f'slot_{slot:02d}.json').get('state')=='finished':continue
   key=('base',slot);attempts[key]=attempts.get(key,0)+1
   if attempts[key]>3:raise RuntimeError('base_tail_QC_retry_limit')
   manual[slot]=start(gpu_step(qc[str(slot)]['job_id'],'dual_crop_qc_worker.py',['--slot',str(slot)]),'qc_tail_base_'+str(slot))
  for slot,p in list(aux.items()):
   if p.poll() is None or load(TAIL/f'slot_{slot:02d}.json').get('state')=='finished':continue
   key=('extra',slot);attempts[key]=attempts.get(key,0)+1
   if attempts[key]>3:raise RuntimeError('extra_tail_QC_retry_limit')
   job=original[str(slot-11)]['allocation_job_id'];aux[slot]=start(gpu_step(job,'extra_qc_tail_worker.py',['--slot',str(slot)]),'qc_tail_extra_'+str(slot))
  if bridge.poll() is not None and load(TAIL/'bridge.json').get('state')!='complete':
   attempts['bridge']=attempts.get('bridge',0)+1
   if attempts['bridge']>3:raise RuntimeError('tail_QC_bridge_retry_limit')
   bridge=start(step(source_job,'bridge_qc_tail.py',['--expected-host',source_host],True),'qc_tail_bridge')
  if not completed and load(TAIL/'bridge.json').get('state')=='complete':
   plan=load(TAIL/'plan.json');plan.update(state='complete',completed_at_utc=now());save(TAIL/'plan.json',plan);completed=True
  base_live=sum(load(ROOT/'independent_qc'/f'slot_{s:02d}.json').get('state')=='reviewing' for s in base)
  extra_live=sum(load(TAIL/f'slot_{s:02d}.json').get('state')=='reviewing' for s in extras)
  save(TAIL/'controller_status.json',{'state':'running','base_reviewers_active':base_live,'additional_reviewers_active':extra_live,'extra_reviews_merged':load(TAIL/'bridge.json').get('copied',0),'redistributed_tasks':load(TAIL/'plan.json').get('task_count'),'updated_at_utc':now()})
  if completed and all(p.poll() is not None for p in list(manual.values())+list(aux.values())+list(controls.values())) and bridge.poll() is not None:
   save(TAIL/'controller_status.json',{'state':'complete','updated_at_utc':now()});return
  time.sleep(5)

if __name__=='__main__':
 try:main()
 except Exception as e:
  with (TAIL/'controller.private_errors.log').open('a') as f:traceback.print_exc(file=f)
  save(TAIL/'controller_status.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()})
  notify('parallel tail review requires attention','Final crop review scheduling requires attention','A local review scheduling step needs attention. Existing GPU workers and allocations remain available. Private diagnostics remain on HPC.')
  while True:time.sleep(30)
