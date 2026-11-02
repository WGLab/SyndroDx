"""Every-minute external monitor; clinical files are never opened."""
import time,json,subprocess,concurrent.futures
from common import ROOT,SLURM,load,save,now,environment,notify
from allocations import active_allocations

def last_metrics():
 p=ROOT/'training/training_metrics.json'
 if not p.exists():return {},None
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-30000));lines=f.read().decode(errors='replace').splitlines()
 for line in reversed(lines):
  try:row=json.loads(line)
  except ValueError:continue
  allowed=['iteration','iter_time','data_time','total_loss','lr','wd','mom','last_layer_lr','dino_local_crops_loss','dino_global_crops_loss','koleo_loss','ibot_loss','backbone_grad_norm','dino_head_grad_norm','ibot_head_grad_norm']
  return {k:row[k] for k in allowed if k in row and isinstance(row[k],(int,float))},p.stat().st_mtime
 return {},p.stat().st_mtime

def sample_one(allocation):
 command=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gpus-per-task=1','--cpu-bind=none','--exact','--mpi=none','/usr/bin/python3',str(ROOT/'gpu_probe.py')]
 try:
  result=subprocess.run(command,env=environment(),capture_output=True,text=True,timeout=15)
  for line in result.stdout.splitlines():
   try:value=json.loads(line)
   except ValueError:continue
   if isinstance(value,dict) and 'node' in value:return dict(value,allocation_job_id=allocation['job_id'])
  return dict(allocation_job_id=allocation['job_id'],state='probe_unavailable')
 except Exception as exc:return dict(allocation_job_id=allocation['job_id'],state='probe_failed',error_type=type(exc).__name__)

def main():
 folder=ROOT/'monitoring';folder.mkdir(exist_ok=True);last_probe=0;last_iter=None;last_change=time.monotonic();last_capacity=-1;previous_phase=None
 while True:
  cycle_started=time.monotonic()
  submit=load(ROOT/'training_submission.json');job=str(submit.get('job_id',''))
  if not job:time.sleep(20);continue
  pool,states=active_allocations();metrics,mtime=last_metrics();phase=load(ROOT/'phase.json');prep=load(ROOT/'preparation_status.json')
  if prep.get('state')=='failed':notify('preparation failed external','Official-code CPT preparation failed','CPU preparation failed before training. GPU dependencies prevent training from starting. Existing models and detailed diagnostics remain on HPC.')
  if phase.get('state')!=previous_phase:previous_phase=phase.get('state');last_change=time.monotonic()
  if metrics.get('iteration')!=last_iter:last_iter=metrics.get('iteration');last_change=time.monotonic()
  if pool and phase.get('state') not in ['waiting_for20_H100_allocations','waiting_for_starting_H100_allocations'] and time.monotonic()-last_probe>=120:
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:tele=list(executor.map(sample_one,pool))
   save(folder/'gpu_telemetry.json',dict(state='sampled',allocations=tele,checked_at_utc=now()));last_probe=time.monotonic()
  if last_capacity==20 and len(pool)<20:
   notify('GPU pool changed','Official-code CPT allocation changed','One or more GPUs in the fixed20-card pool are no longer running. The monitor has not cancelled other reservations or changed the training recipe. Private status remains on HPC.')
  last_capacity=len(pool)
  q=subprocess.run([str(SLURM/'squeue'),'-h','-j',job,'-o','%T|%R'],env=environment(),capture_output=True,text=True,timeout=25)
  state=q.stdout.strip().split('|')[0] if q.stdout.strip() else 'not_in_queue'
  if time.monotonic()-last_change>900 and phase.get('state') in ['training','training_initializing']:
   recent_io=any(time.time()-p.stat().st_mtime<300 for p in (ROOT/'training/ckpt').glob('*/*') if p.is_file())
   if not recent_io:notify('stale progress '+job,'Official-code CPT progress requires attention','The external monitor detected slow or stalled progress. It has not killed allocations or changed the recipe. Private logs and checkpoints remain on HPC.')
  save(folder/'status.json',dict(state=state,controller_job_id=job,running_H100_allocations=len(pool),required_H100_allocations=20,phase=phase,preparation_state=prep.get('state'),metrics=metrics,last_metrics_file_mtime=mtime,monitor_interval_seconds=60,GPU_sampling_interval_seconds=120,checked_at_utc=now()))
  if state=='not_in_queue':
   acct=subprocess.run([str(SLURM/'sacct'),'-nP','-j',job,'-X','--format=State,ExitCode'],env=environment(),capture_output=True,text=True,timeout=25).stdout.strip()
   terminal=acct.split('|')[0] if acct else ''
   if terminal.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):
    notify('controller termination '+job,'Official-code CPT controller ended','The scheduler reports that the training controller ended before ordinary completion. All existing checkpoint files are retained. No automatic recipe change was made; private diagnostics remain on HPC.');return
   if terminal.startswith('COMPLETED'):
    save(folder/'status.json',dict(state='training_complete_allocations_retained',running_H100_allocations=len(pool),phase=phase,checked_at_utc=now()));return
  time.sleep(max(1,60-(time.monotonic()-cycle_started)))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'monitoring/failure.json',dict(state='failed',error_type=type(exc).__name__,updated_at_utc=now()))
  notify('external monitor failure','Official-code CPT monitor requires attention','The external monitor encountered an error. This does not imply training stopped. Private diagnostics remain on HPC.')
  raise SystemExit(1)
