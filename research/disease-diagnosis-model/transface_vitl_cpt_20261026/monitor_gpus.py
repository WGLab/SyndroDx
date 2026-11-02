"""Sample utilization of this run's allocated GPUs; aggregate operational output only."""
import concurrent.futures
import fcntl
import os
import statistics
import subprocess
import time
from control import WORK,SLURM,load,save,now,environment,alive

def sample(allocation):
    args=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--job-name=transface-monitor',
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=128M','--gpus-per-task=1',
        '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
        '/usr/bin/timeout','6s','nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits','--loop-ms=1000']
    try:
        result=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=25)
        rows=[]
        for line in result.stdout.splitlines():
            fields=line.split(',')
            if len(fields)==3:
                try:rows.append(tuple(float(v.strip()) for v in fields))
                except ValueError:pass
        if not rows:return dict(job_id=allocation['job_id'],state='unavailable')
        return dict(job_id=allocation['job_id'],node=allocation['node'],state='sampled',samples=len(rows),
            gpu_util_mean_percent=statistics.mean(r[0] for r in rows),gpu_util_max_percent=max(r[0] for r in rows),
            memory_used_mib_max=max(r[1] for r in rows),memory_total_mib=rows[0][2])
    except (subprocess.SubprocessError,OSError):return dict(job_id=allocation['job_id'],state='unavailable')

def main():
    os.umask(0o077)
    with (WORK/'gpu_monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        misses=0
        while misses<5:
            leader=load(WORK/'supervisor_process.json')
            misses=0 if alive(leader) else misses+1
            pool=load(WORK/'active_pool.json').get('allocations',[])
            if pool:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(20,len(pool))) as executor:
                    readings=list(executor.map(sample,pool))
                report=dict(state='sampled',gpus=readings,training_state=load(WORK/'status.json').get('state'),
                    completed_steps=load(WORK/'status.json').get('completed_steps'),updated_at_utc=now())
                save(WORK/'gpu_utilization.json',report)
                import json
                with (WORK/'gpu_utilization.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
            time.sleep(60)

if __name__=='__main__':main()
