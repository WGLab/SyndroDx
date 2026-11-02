"""Persistent pilot controller using existing free allocations only."""
import fcntl
import subprocess
import time
from common import *

def launch(job,script,args,name):
    command=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1',
        '--cpus-per-task=4','--mem=48G','--gpus-per-task=1','--overlap','--exact','--immediate=30','--mpi=none',
        str(PYTHON),str(WORK/script),*args]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:
        return subprocess.Popen(command,env=environment(),stdout=f,stderr=f)

def main():
    os.umask(0o077);lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    assert load(WORK/'tests.json')['state']=='passed' and load(WORK/'data_receipt.json')['state']=='passed'
    p=load(WORK/'protocol.json');cases={f'{c}__seed{s}':(c,str(s)) for s in p['seeds'] for c in p['conditions']}
    active={};retired={};attempts={};waiting={}
    while True:
        for key,item in list(active.items()):
            rc=item['process'].poll()
            if rc is not None:
                del active[key];retired[key]=time.monotonic()
                if rc==0:waiting[key]=time.monotonic()
                elif attempts[key]>=3:raise RuntimeError('repeated_worker_failure')
        smoke=load(WORK/'smoke.json').get('state')=='passed'
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        for key,t in list(waiting.items()):
            if (key=='smoke' and smoke) or key in done:del waiting[key]
            elif time.monotonic()-t>600:raise RuntimeError('worker_receipt_not_visible')
        try:jobs=free_jobs()
        except Exception:jobs=[]
        busy={v['job'] for v in active.values()};jobs=[j for j in jobs if j['id'] not in busy]
        tasks=[('smoke','smoke.py',[])] if not smoke else [(n,'train.py',list(cases[n])) for n in cases if n not in done]
        for key,script,args in tasks:
            if not jobs or len(active)>=4:break
            if key in active or key in waiting or time.monotonic()-retired.get(key,-1e9)<90:continue
            failure=load(WORK/'smoke_failure.json') if key=='smoke' else load(WORK/'cases'/key/'failure.json')
            if failure.get('error_type') in ['FloatingPointError','AssertionError','OutOfMemoryError']:raise RuntimeError('worker_validation_failed')
            job=jobs.pop(0);attempts[key]=attempts.get(key,0)+1
            active[key]=dict(job=job['id'],process=launch(job,script,args,'BLIP2-stage1-'+key))
        save(WORK/'status.json',dict(state='training' if smoke else 'waiting_or_GPU_smoke',completed=len(done),total=len(cases),
            active={k:v['job'] for k,v in active.items()},updated_at_utc=now()))
        time.sleep(10)
    with (WORK/'logs/summary.private.log').open('ab') as f:
        subprocess.run([str(PYTHON),str(WORK/'summarize.py')],env=environment(),stdout=f,stderr=f,check=True)
    deadline=time.monotonic()+600
    while load(WORK/'results.json').get('state')!='complete':
        if time.monotonic()>deadline:raise RuntimeError('summary_receipt_not_visible')
        time.sleep(10)
    save(WORK/'status.json',dict(state='complete',completed=len(cases),total=len(cases),allocations_preserved=True,updated_at_utc=now()))
    try:
        notify('BLIP-2 Stage1 pilot complete',json.dumps(load(WORK/'results.json'),indent=2))
        save(WORK/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:save(WORK/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('BLIP-2 Stage1 requires inspection','Local Stage1 pilot controller stopped: '+type(e).__name__+'. Frozen baseline and GPU allocations are retained.')
        except Exception:pass
        raise
