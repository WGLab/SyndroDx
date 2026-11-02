"""Run local feature extraction then matched ablations without releasing allocations."""
import fcntl
import subprocess
import time
from common import *


def launch(job, script, args, name):
    cmd = [str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1',
           '--cpus-per-task=4','--mem=48G','--gpus-per-task=1','--overlap','--exact','--immediate=30','--mpi=none',
           str(PYTHON),str(WORK/script),*args]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:
        return subprocess.Popen(cmd,env=environment(),stdout=f,stderr=f)


def main():
    os.umask(0o077)
    lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    assert load(WORK/'data_receipt.json')['state']=='weak_patient_association_release'
    cases=load(WORK/'cases.json')['cases']
    active,retired,attempts={},{},{}
    while True:
        for key,item in list(active.items()):
            rc=item['process'].poll()
            if rc is not None:
                del active[key];retired[key]=time.monotonic()
                if rc and attempts[key]>=3:raise RuntimeError('repeated_local_worker_failure')
        features=load(WORK/'features/status.json').get('state')=='complete'
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        try:jobs=free_jobs()
        except Exception:jobs=[]
        busy={v['job'] for v in active.values()}
        jobs=[j for j in jobs if j['id'] not in busy]
        tasks=[('extract','extract.py',[])] if not features else [(n,'train.py',[n]) for n in cases if n not in done]
        for key,script,args in tasks:
            if not jobs or len(active)>=2:break
            if key in active or time.monotonic()-retired.get(key,-1e9)<90:continue
            job=jobs.pop(0);attempts[key]=attempts.get(key,0)+1
            active[key]=dict(job=job['id'],process=launch(job,script,args,'CHOP-pair-'+key))
        save(WORK/'status.json',dict(state='training' if features else 'waiting_or_extracting_features',
            completed=len(done),total=len(cases),active={k:v['job'] for k,v in active.items()},updated_at_utc=now()))
        time.sleep(10)
    save(WORK/'status.json',dict(state='summarizing',completed=len(cases),total=len(cases),updated_at_utc=now()))
    with (WORK/'logs/statistics.private.log').open('ab') as f:
        subprocess.run([str(PYTHON),str(WORK/'summarize.py')],env=environment(),stdout=f,stderr=f,check=True)
    save(WORK/'status.json',dict(state='complete',completed=len(cases),total=len(cases),updated_at_utc=now()))
    try:
        notify('CHOP paired-data ablation complete',json.dumps(load(WORK/'results.json'),indent=2))
        save(WORK/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:
        save(WORK/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
