"""Read-only GPU telemetry on owned allocations; no patient data accessed."""
import concurrent.futures,subprocess,statistics
from runtime import *
from controller import jobs
def probe(j):
    cmd=[str(SLURM/'srun'),'--jobid='+j['id'],'--job-name=cranio-qwen-monitor','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gpus-per-task=1','--overlap','--exact','--mpi=none',
        'nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']
    try:
        result=subprocess.run(cmd,env=env(),capture_output=True,text=True,timeout=25,check=True)
        values=[float(v.strip()) for v in result.stdout.strip().splitlines()[0].split(',')]
        return dict(job_id=j['id'],node=j['node'],used_MiB=values[0],total_MiB=values[1],utilization=values[2])
    except Exception as e:return dict(job_id=j['id'],error_type=type(e).__name__)
def main():
    pool=jobs()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(probe,pool))
    good=[r for r in rows if 'utilization' in r]
    summary=dict(allocations=len(pool),sampled=len(good),nonzero_memory_cards=sum(r['used_MiB']>1000 for r in good),
        utilization_mean=round(statistics.mean(r['utilization'] for r in good),1) if good else None,
        utilization_min=min((r['utilization'] for r in good),default=None),utilization_max=max((r['utilization'] for r in good),default=None),
        mean_used_GiB=round(statistics.mean(r['used_MiB']/1024 for r in good),1) if good else None,updated_at_utc=now())
    save(WORK/'gpu_probe.json',dict(summary=summary,cards=rows));print(__import__('json').dumps(summary))
if __name__=='__main__':main()
