import concurrent.futures
import statistics
import subprocess
from common import WORK,SLURM,environment,save,now

def sample(job):
    args=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--job-name=gmdb-s2-monitor','--nodes=1','--ntasks=1',
        '--cpus-per-task=1','--mem=128M','--gpus-per-task=1','--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
        '/usr/bin/timeout','4s','nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits','--loop-ms=1000']
    try:
        result=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=20);rows=[]
        for line in result.stdout.splitlines():
            try:
                v=[float(x) for x in line.split(',')]
                if len(v)==3:rows.append(v)
            except ValueError:pass
        if rows:return dict(job_id=job['job_id'],samples=len(rows),gpu_util_mean_percent=statistics.mean(v[0] for v in rows),memory_used_mib=max(v[1] for v in rows))
    except (OSError,subprocess.SubprocessError):pass
    return dict(job_id=job['job_id'],state='unavailable')

def sample_pool(pool):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pool)) as executor:rows=list(executor.map(sample,pool))
    save(WORK/'gpu_utilization.json',dict(gpus=rows,updated_at_utc=now()))
