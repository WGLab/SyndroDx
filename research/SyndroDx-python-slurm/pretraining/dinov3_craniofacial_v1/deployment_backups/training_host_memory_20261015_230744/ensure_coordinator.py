"""Restart only the pipeline coordinator if it is absent; preserve all GPU jobs."""
import fcntl,subprocess
from runtime import ROOT,RUN,SLURM,load,save,now

def main():
 if load(RUN/'ssl_training_completion.json').get('state')=='ssl_training_complete':return
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
