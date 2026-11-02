"""Activate the validated coordinator without interrupting GPU training."""
import os
import signal
import socket
import subprocess
import sys
import time
from common import WORK,STAGE1,PYTHON,SLURM,load,save,environment,now,verify_code

def main():
    os.umask(0o077);verify_code()
    for name in ('tests_receipt.json','smoke/receipt.json','patient_label_preflight.json'):
        if load(WORK/name).get('state')!='passed':raise RuntimeError('activation_preflight_incomplete')
    sys.path.insert(0,str(STAGE1))
    from control import identity
    old=load(STAGE1/'supervisor_process.json')
    before=load(STAGE1/'controller_process.json')
    pid=old.get('pid')
    if pid and identity(pid)==old.get('start_identity'):
        if old.get('host')!=socket.gethostname():raise RuntimeError('supervisor_host_mismatch')
        path=f'/proc/{pid}'
        with open(path+'/cmdline','rb') as stream:arguments=stream.read().split(b'\0')
        if os.stat(path).st_uid!=os.getuid() or str(STAGE1/'supervisor.py').encode() not in arguments:
            raise RuntimeError('supervisor_identity_mismatch')
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+20
        while identity(pid)==old['start_identity'] and time.monotonic()<deadline:time.sleep(.2)
        if identity(pid)==old['start_identity']:raise RuntimeError('supervisor_did_not_exit')
    with (STAGE1/'logs/supervisor-follow-on.private.log').open('ab') as output:
        child=subprocess.Popen([str(PYTHON),str(STAGE1/'supervisor.py')],env=environment(),stdout=output,stderr=output,stdin=subprocess.DEVNULL,start_new_session=True)
    save(STAGE1/'supervisor_process.json',dict(pid=child.pid,start_identity=identity(child.pid),host=socket.gethostname(),updated_at_utc=now()))
    time.sleep(3)
    if child.poll() is not None:raise RuntimeError('upgraded_supervisor_failed')
    after=load(STAGE1/'controller_process.json')
    if before!=after:raise RuntimeError('training_controller_unexpectedly_changed')
    name='cranio-automatic-follow-on'
    existing=subprocess.run([str(SLURM/'squeue'),'-h','-u','research_user','--name='+name,'-o','%i'],env=environment(),capture_output=True,text=True,check=True,timeout=30).stdout.split()
    if len(existing)>1:raise RuntimeError('duplicate_workflow_jobs')
    if existing:job=existing[0]
    else:
        job=subprocess.run([str(SLURM/'sbatch'),'--parsable',str(WORK/'workflow.sbatch')],env=environment(),capture_output=True,text=True,check=True,timeout=60).stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('invalid_workflow_submission_receipt')
    save(WORK/'workflow_job.json',dict(job_id=job,name=name,submitted_at_utc=now()))
    save(WORK/'activation_receipt.json',dict(state='submitted',workflow_job=job,training_controller_unchanged=True,
        gpu_allocations_cancelled=[],updated_at_utc=now()))
    print('Automatic coordinator submitted:',job,'; GPU training controller preserved')

if __name__=='__main__':main()
