"""Aggregate telemetry for experiment-owned GPU workers only."""
import subprocess,concurrent.futures,json,sys
from matched_runtime import *
REMOTE = '''import json,os,subprocess,torch
p=torch.cuda.get_device_properties(0)
ident=str(getattr(p,"uuid",os.environ.get("CUDA_VISIBLE_DEVICES","0")))
if "-" in ident and not ident.startswith("GPU-"):ident="GPU-"+ident
v=subprocess.run(["nvidia-smi","-i",ident,"--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip().split(",")
print(json.dumps(dict(utilization_percent=float(v[0]),memory_used_mib=float(v[1]),memory_total_mib=float(v[2]))))
'''
def one(arg):
    arm,job=arg
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name=objective-telemetry','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=2G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),'-c',REMOTE]
    result=subprocess.run(cmd,env=environment(),capture_output=True,text=True,timeout=45)
    if result.returncode:return dict(arm=arm,job=job['id'],state='telemetry_unavailable')
    return dict(arm=arm,job=job['id'],state='complete',**json.loads(result.stdout.strip().splitlines()[-1]))
if __name__=='__main__':
    args=[(arm,j) for arm in ARMS for j in load(ROOT/arm/'active_pool.json').get('jobs',[])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:rows=list(ex.map(one,args))
    save(ROOT/'gpu_snapshot.json',dict(gpus=rows,updated_at_utc=now()))
    print(json.dumps(rows))
