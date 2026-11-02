"""A single GPU does useful prework, then participates in the shared model."""
import os
import signal
import subprocess
import sys
import time
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS
from common import save,load,now,environment,PYTHON


def main():
    slot=int(sys.argv[1]);job=os.environ['SLURM_JOB_ID']
    deadline=time.monotonic()+600
    while True:
        registry=load(WORK/'pool_registry.json')
        if registry.get('jobs',{}).get(str(slot))==job:break
        if not (WORK/'submission_in_progress.json').exists() or time.monotonic()>deadline:
            raise RuntimeError('single_gpu_lease_identity_mismatch')
        time.sleep(2)
    path=WORK/'leases'/f'{job}.json';path.parent.mkdir(exist_ok=True)
    env=environment();env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
    node=os.environ['SLURMD_NODENAME']
    def receipt(state,**extra):
        save(path,dict(state=state,job_id=job,slot=slot,node=node,updated_at_utc=now(),**extra))
    signal.signal(signal.SIGUSR1,lambda *_:save(WORK/'stop_requested.json',dict(reason='lease_walltime_warning',job_id=job,updated_at_utc=now())))
    if (WORK/'pool_done.json').exists():return
    receipt('diagnostic_prework')
    proc=None
    try:
        with (WORK/'leases'/f'{job}.staging.private.log').open('ab') as stream:
            proc=subprocess.Popen([str(PYTHON),str(WORK/'staging_worker.py')],env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=stream)
        join_started=None;diagnostic_error=False
        while True:
            if (WORK/'pool_done.json').exists():break
            joining=(WORK/'join_requested.json').exists()
            if joining:
                join_started=join_started or time.monotonic()
                if proc.poll() is None and time.monotonic()-join_started>300:
                    proc.terminate()
            if proc.poll() is not None:
                diagnostic_error=proc.returncode!=0 and not joining
                receipt('available',diagnostic_error=diagnostic_error,
                    diagnostic_complete=not joining and proc.returncode==0)
            else:receipt('yielding_for_training' if joining else 'diagnostic_prework')
            heartbeat=WORK/'supervisor_heartbeat.json'
            if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>30*60:
                raise RuntimeError('persistent_supervisor_heartbeat_expired')
            time.sleep(10)
        receipt('pool_finished')
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
