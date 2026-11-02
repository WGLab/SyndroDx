"""Lightweight local launcher; waits for an allocated CPU then hands off."""
import fcntl
import os
import subprocess
import time
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS
from common import load,save,now,environment,SLURM,PYTHON,notify
from batch_job import preflight


def status(state,**extra):
    save(WORK/'launch_status.json',dict(state=state,updated_at_utc=now(),**extra))


def main():
    os.umask(0o077);preflight()
    with (WORK/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry=load(WORK/'pool_registry.json');ids=list(registry['jobs'].values())
        if len(ids)!=20 or len(set(ids))!=20:raise RuntimeError('twenty_unique_jobs_required')
        deadline=time.monotonic()+9*24*3600
        while time.monotonic()<deadline:
            try:
                result=subprocess.run([str(SLURM/'squeue'),'-h','-j',','.join(ids),'-o','%i|%T|%N'],
                    capture_output=True,text=True,timeout=15,check=True)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                status('scheduler_temporarily_unavailable_retrying');time.sleep(30);continue
            ready=[];pending=0
            for line in result.stdout.splitlines():
                job,state,node=line.strip().split('|')
                if job not in ids:continue
                if state=='RUNNING' and load(WORK/'leases'/f'{job}.json').get('job_id')==job:ready.append(job)
                if state in ('PENDING','CONFIGURING'):pending+=1
            status('waiting_for_first_registered_allocation',registered_running_gpus=len(ready),pending_single_gpu_jobs=pending)
            if not ready:time.sleep(10);continue
            job=sorted(ready,key=int)[0]
            env=environment();env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
            env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
            args=[str(SLURM/'srun'),'--jobid='+job,'--job-name=cranio-convergence-coordinator',
                '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G','--gres=none',
                '--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'pool_controller.py')]
            with (WORK/'embedded_controller.private.log').open('ab') as stream:
                proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
            timeout=time.monotonic()+180
            while proc.poll() is None and time.monotonic()<timeout:
                heartbeat=load(WORK/'controller_heartbeat.json')
                if heartbeat:
                    status('handed_to_incremental_controller',coordinator_parent_job=job,srun_client_pid=proc.pid)
                    return
                time.sleep(5)
            raise RuntimeError('coordinator_startup_requires_inspection')
        raise RuntimeError('queue_wait_window_expired')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('launcher_requires_inspection',error_type=type(error).__name__)
        notify('convergence launcher failure','Craniofacial continuation launcher needs inspection',
            'The local launcher stopped on a scheduler or integrity check. Preserved checkpoints remain on HPC; no private data is included in this notice.')
        raise
