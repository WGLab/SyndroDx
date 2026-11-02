"""Interrupt only the CPU launcher, and only after a durable official checkpoint."""
import os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,'/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
from common import ROOT,SLURM,load,save,now,environment,notify

def checkpoint_ready(folder,world):
 metadata=folder/'.metadata';shards=list(folder.glob('*.distcp')) if folder.is_dir() else []
 return metadata.is_file() and metadata.stat().st_size>0 and len(shards)>=world and all(p.stat().st_size>0 for p in shards)

def state(job):
 r=subprocess.run([str(SLURM/'squeue'),'-h','-j',str(job),'-o','%u|%j|%T'],env=environment(),capture_output=True,text=True,timeout=25,check=True)
 return r.stdout.strip()

def main():
 os.umask(0o077);deadline=time.monotonic()+3600
 while time.monotonic()<deadline:
  request=load(ROOT/'batch_resize_request.json')
  if not request.get('handoff_ready'):time.sleep(5);continue
  old=request['previous_controller_job_id'];new=request['new_controller_job_id']
  assert old!=new and str(old).isdigit() and str(new).isdigit()
  checkpoint=ROOT/'training/ckpt'/str(request['target_checkpoint_iteration'])
  if checkpoint_ready(checkpoint,request['expected_world_size']):
   s=state(old)
   if s:
    assert s=='research_user|official-cpt-v2-controller|RUNNING',s
    save(ROOT/'batch_gate_status.json',dict(state='committed_checkpoint_verified_signalling_CPU_controller',checkpoint_iteration=request['target_checkpoint_iteration'],old_controller=old,new_controller=new,GPU_parent_allocations_cancelled=False,updated_at_utc=now()))
    subprocess.run([str(SLURM/'scancel'),'--signal=INT','--batch',str(old)],env=environment(),check=True)
    # SIGINT permits Python finally blocks to stop only the launcher's own steps.
    for _ in range(24):
     if not state(old):break
     time.sleep(5)
    else:
     # If batch SIGINT is ignored, stop only the identified project training steps.
     # Their controller then cleans up and exits; none of the parent GPU leases ends.
     active=load(ROOT/'active_clients.json');owned=set(active.get('allocation_job_ids',[]))
     owned.update(x['lookup_id'] for x in load(ROOT/'selected_pool.json').get('allocations',[]) if x.get('job_id') in owned and x.get('lookup_id'))
     q=subprocess.run([str(SLURM/'squeue'),'-h','--steps','-u','research_user','-o','%i|%C'],env=environment(),capture_output=True,text=True,check=True,timeout=25)
     targets=[]
     for line in q.stdout.splitlines():
      fields=line.split('|')
      if len(fields)!=2:continue
      jobstep,cpus=fields
      if jobstep.split('.')[0] in owned and jobstep.rsplit('.',1)[-1].isdigit() and cpus.strip()=='4':targets.append(jobstep)
     if not targets:raise RuntimeError('no_owned_training_steps_found_for_handoff')
     subprocess.run([str(SLURM/'scancel'),'--signal=TERM',*targets],env=environment(),check=True)
   save(ROOT/'batch_gate_status.json',dict(state='checkpoint_handoff_requested',checkpoint_iteration=request['target_checkpoint_iteration'],old_controller=old,new_controller=new,GPU_parent_allocations_cancelled=False,updated_at_utc=now()))
   return
  if not state(old):raise RuntimeError('original_controller_ended_before_requested_checkpoint')
  time.sleep(5)
 raise RuntimeError('checkpoint_gate_timeout')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'batch_gate_status.json',dict(state='failed',error_type=type(exc).__name__,GPU_parent_allocations_cancelled=False,updated_at_utc=now()))
  notify('batch checkpoint gate failure','CPT batch change requires attention','The requested batch handoff did not complete. The original encoder and checkpoints are preserved. GPU reservations were not cancelled; private details remain on HPC.')
  raise SystemExit(1)
