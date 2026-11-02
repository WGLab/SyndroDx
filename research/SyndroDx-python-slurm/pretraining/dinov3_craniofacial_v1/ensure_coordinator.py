"""Restart only the pipeline coordinator if it is absent; preserve all GPU jobs."""
import datetime,fcntl,subprocess,time
from runtime import ROOT,RUN,SLURM,load,save,now,notify

def training_health():
 launcher=load(ROOT/'training/launcher.json');state=load(ROOT/'training/state.json')
 handoff=load(ROOT/'training/checkpoint_batch_handoff.json')
 try:handoff_age=time.time()-datetime.datetime.fromisoformat(handoff.get('updated_at_utc','')).timestamp()
 except ValueError:handoff_age=999999
 if handoff.get('state') in ['applying','restart_enabled'] and handoff_age<300:
  save(ROOT/'training/health_status.json',{'state':'planned_checkpoint_batch_handoff','checked_at_utc':now()});return
 status='healthy';reason=None
 if launcher.get('state')=='attention_required':
  status='attention_required';reason='training_launch_or_rank_failed'
  event='SSL training failure '+str(launcher.get('updated_at_utc','unknown_attempt'))
 elif launcher.get('state')=='running' and state.get('state') in ['training','initializing']:
  try:age=time.time()-datetime.datetime.fromisoformat(state['updated_at_utc']).timestamp()
  except (KeyError,ValueError):age=0
  threshold=900 if state.get('state')=='training' else 1800
  if age>threshold:
   status='stale_progress';reason='training_progress_not_advancing'
   event='SSL training stale '+str(launcher.get('started_at_utc','unknown_attempt'))+' '+str(state.get('iteration','initializing'))
 if reason:
  notify(event,'SSL training requires attention','The local SSL training run has failed or is not advancing within its monitoring window. Existing GPU reservations and all clinical data are retained on HPC. Check the local training status for recovery diagnostics.')
 save(ROOT/'training/health_status.json',{'state':status,'reason':reason,'checked_at_utc':now()})

def main():
 from ensure_checkpoint_batch_watch import ensure as ensure_batch_watch
 ensure_batch_watch()
 from ensure_pipeline import ensure
 ensure()
 if load(RUN/'ssl_training_completion.json').get('state')=='ssl_training_complete':return
 training_health()
 lock=(ROOT/'coordinator_submission.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 name='craniofacial-ssl-auto'
 r=subprocess.run([str(SLURM/'squeue'),'-h','-u','research_user','-n',name,'-o','%i|%T'],capture_output=True,text=True,timeout=25,check=True)
 if r.stdout.strip():return
 body=ROOT/'coordinator.sbatch'
 body.write_text('#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true DO_NOT_TRACK=1\nexec '+str(ROOT/'venv/bin/python')+' '+str(ROOT/'controller.py')+' --loop\n')
 r=subprocess.run([str(SLURM/'sbatch'),'--parsable','--partition=defq','--job-name='+name,'--cpus-per-task=1','--mem=4G','--time=20-00:00:00','--chdir='+str(ROOT),'--output='+str(ROOT/'coordinator.slurm.private.log'),str(body)],capture_output=True,text=True,timeout=30,check=True)
 job=r.stdout.strip().split(';')[0]
 if not job.isdigit():raise RuntimeError('coordinator_submission_failed')
 save(ROOT/'coordinator_submission.json',{'state':'submitted','job_id':job,'submitted_at_utc':now()})
 print({'coordinator_job_id':job,'state':'submitted'})
if __name__=='__main__':main()
