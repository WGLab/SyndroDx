"""Reuse completed primary allocations by splitting unstarted shared crop tasks."""
import fcntl,json,subprocess,time,traceback,sqlite3
from runtime import ROOT,RUN,load,save,now,scheduler,claim_jobs,notify
from crop_delegation import ACCEL,PIPELINE
from accelerate_crops import start,step

def gpu_step(job,script,args):
 cmd=step(job,script,args)
 cmd.remove('--exclusive');cmd.insert(1,'--overlap')
 return cmd

def pending_count(slot):
 path=ACCEL/f'worker_{slot:02d}.tasks.jsonl'
 with path.open() as f:keys={t['key'] for t in map(json.loads,f)}
 db=sqlite3.connect((ACCEL/f'results_{slot:02d}.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
 try:done={r[0] for r in db.execute('SELECT key FROM results')}
 finally:db.close()
 return len(keys-done)

def main():
 lock=(ACCEL/'rebalancer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 registry=load(ACCEL/'rebalancing.json',{'extras':{},'source_transfers':{}})
 if registry['extras']:raise RuntimeError('existing_rebalance_requires_adoption')
 original=load(PIPELINE/'handoff_launches.json');gpu={};bridges={};locks={};attempts={};source_time={};qc_restarts={};next_slot=max([10]+[int(p.name.split('_')[1].split('.')[0]) for p in ACCEL.glob('extra_*.transfer.json')])+1
 while True:
  jobs,busy=scheduler();qc=load(ROOT/'independent_qc/launches.json');plan=load(ACCEL/'plan.json')
  for extra,p in list(bridges.items()):
   if p.poll() is None or load(ACCEL/f'extra_{extra:02d}.bridge.json').get('state')=='complete':continue
   entry=registry['extras'][str(extra)];attempts[('bridge',extra)]=attempts.get(('bridge',extra),0)+1
   if attempts[('bridge',extra)]>3:raise RuntimeError('extra_result_bridge_retry_limit')
   bridges[extra]=start(step(entry['source_job'],'bridge_extra_results.py',['--extra',str(extra),'--expected-host',entry['source_host']],True),'bridge_extra_'+str(extra))
  for extra,p in list(gpu.items()):
   if p.poll() is None or load(ACCEL/f'worker_{extra:02d}.json').get('state')=='primary_crop_partition_complete':continue
   entry=registry['extras'][str(extra)];job=entry['job_id']
   attempts[('gpu',extra)]=attempts.get(('gpu',extra),0)+1
   if attempts[('gpu',extra)]>3:raise RuntimeError('extra_crop_worker_retry_limit')
   gpu[extra]=start(gpu_step(job,'extra_crop_worker.py',['--slot',str(extra)]),'extra_gpu_'+str(extra));busy.add(job)
  for source,p in list(qc_restarts.items()):
   if p.poll() is None or load(ROOT/'independent_qc'/f'slot_{source:02d}.json').get('state')=='finished':continue
   key=('qc_restart',source);attempts[key]=attempts.get(key,0)+1
   if attempts[key]>3:raise RuntimeError('rebalanced_QC_restart_retry_limit')
   qc_restarts[source]=start(gpu_step(qc[str(source)]['job_id'],'dual_crop_qc_worker.py',['--slot',str(source)]),'shared_recovery_'+str(source))
  if plan.get('state')!='complete':
   for primary in range(6):
    job=original[str(primary)]['allocation_job_id']
    if jobs.get(job,{}).get('state')!='RUNNING':continue
    if load(PIPELINE/'workers'/f'slot_{primary:02d}.json').get('state')!='finished_provisional':continue
    if any(registry['extras'][str(e)]['job_id']==job and p.poll() is None for e,p in gpu.items()):continue
    candidates=[]
    for source in range(11):
     state=load(ROOT/'independent_qc'/f'slot_{source:02d}.json')
     if state.get('state') not in ['assisting_primary_crop','reviewing']:continue
     if registry['source_transfers'].get(str(source),0)>=2 or time.time()-source_time.get(source,0)<180:continue
     try:n=pending_count(source)
     except (sqlite3.Error,OSError):continue
     if n>=500:candidates.append((n,source))
    if not candidates:continue
    if job not in locks:
     try:locks[job]=claim_jobs([job])
     except BlockingIOError:continue
    _,source=max(candidates);source_job=qc[str(source)]['job_id'];source_host=jobs[source_job]['node'];extra=next_slot;next_slot+=1
    entry={'job_id':job,'original_primary_slot':primary,'source_slot':source,'source_job':source_job,'source_host':source_host,'state':'preparing_transfer','at_utc':now()};registry['extras'][str(extra)]=entry;save(ACCEL/'rebalancing.json',registry)
    p=start(step(source_job,'split_crop_work.py',['--source',str(source),'--extra',str(extra),'--expected-host',source_host],True),'split_extra_'+str(extra))
    receipt={}
    for _ in range(300):
     receipt=load(ACCEL/f'extra_{extra:02d}.transfer.json')
     if receipt.get('state')=='sealed' or p.poll() is not None:break
     time.sleep(.2)
    if receipt.get('state')!='sealed':
     entry['state']='transfer_not_started';save(ACCEL/'rebalancing.json',registry);continue
    current=load(ROOT/'independent_qc/launches.json');current[str(source)].update(launched_at=time.time(),mode='rebalanced_shared_crop_QC');save(ROOT/'independent_qc/launches.json',current)
    qc_restarts[source]=start(gpu_step(source_job,'dual_crop_qc_worker.py',['--slot',str(source)]),'shared_restart_'+str(extra))
    if p.wait()!=0:raise RuntimeError('split_checkpoint_handoff_failed')
    entry.update(state='running',task_count=receipt['task_count']);registry['source_transfers'][str(source)]=registry['source_transfers'].get(str(source),0)+1;save(ACCEL/'rebalancing.json',registry);source_time[source]=time.time()
    bridges[extra]=start(step(source_job,'bridge_extra_results.py',['--extra',str(extra),'--expected-host',source_host],True),'bridge_extra_'+str(extra))
    gpu[extra]=start(gpu_step(job,'extra_crop_worker.py',['--slot',str(extra)]),'extra_gpu_'+str(extra));busy.add(job)
  live=sum(p.poll() is None for p in gpu.values());save(ACCEL/'rebalancer_status.json',{'state':'running','additional_original_GPUs_busy':live,'extra_partitions':len(gpu),'updated_at_utc':now()})
  if plan.get('state')=='complete' and all(p.poll() is not None for p in gpu.values()) and all(p.poll() is not None for p in bridges.values()) and all(p.poll() is not None for p in qc_restarts.values()):
   save(ACCEL/'rebalancer_status.json',{'state':'complete','updated_at_utc':now()});return
  time.sleep(10)

if __name__=='__main__':
 try:main()
 except Exception as e:
  with (ACCEL/'rebalancer.private_errors.log').open('a') as f:traceback.print_exc(file=f)
  save(ACCEL/'rebalancer_status.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()})
  notify('crop rebalancing needs attention','Crop worker scheduling requires attention','A local crop scheduling step needs attention. Existing GPU workers and reservations are retained. Private diagnostics remain on HPC.')
  while True:time.sleep(30)
