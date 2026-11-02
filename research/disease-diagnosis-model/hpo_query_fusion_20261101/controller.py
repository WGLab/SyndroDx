"""Persistent local orchestration. Preserve all salloc allocations and checkpoints."""
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

def restore_background():
    r=load(WORK/'priority_yield.json')
    if r:
        flag=Path(r['flag'])
        if load(flag).get('owner')==str(WORK):
            flag.unlink()
            save(WORK/'priority_restored.json',dict(state='background_seed_resume_allowed',allocations_preserved=True,updated_at_utc=now()))

def main():
    os.umask(0o077)
    lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    assert load(WORK/'data_receipt.json')['state']=='passed'
    assert load(WORK/'tests.json')['state']=='passed'
    p=load(WORK/'protocol.json')
    # First finish one seed across all modes, then repeat the prespecified seed set.
    cases={f'{mode}__seed{seed}':(mode,str(seed)) for seed in p['seeds'] for mode in p['modes']}
    active={};retired={};attempts={};receipt_wait={}
    while True:
        for key,record in list(active.items()):
            rc=record['process'].poll()
            if rc is not None:
                del active[key];retired[key]=time.monotonic()
                if rc==0:receipt_wait[key]=time.monotonic()
                elif attempts[key]>=3:raise RuntimeError('repeated_worker_failure')
        features=load(WORK/'features/receipt.json').get('state')=='complete'
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        for key,t in list(receipt_wait.items()):
            if (key=='extract' and features) or key in done:del receipt_wait[key]
            elif time.monotonic()-t>600:raise RuntimeError('worker_receipt_not_visible')
        try:jobs=free_jobs()
        except Exception:jobs=[]
        busy={v['job'] for v in active.values()};jobs=[j for j in jobs if j['id'] not in busy]
        tasks=[('extract','extract.py',[])] if not features else [(n,'train.py',list(cases[n])) for n in cases if n not in done]
        for key,script,args in tasks:
            if not jobs or len(active)>=2:break
            if key in active or key in receipt_wait or time.monotonic()-retired.get(key,-1e9)<90:continue
            # Numerical failures require inspection; do not blindly retry them.
            failure=load(WORK/'features/failure.json') if key=='extract' else load(WORK/'cases'/key/'failure.json')
            if failure.get('error_type') in ['FloatingPointError','AssertionError']:raise RuntimeError('worker_validation_failed')
            job=jobs.pop(0);attempts[key]=attempts.get(key,0)+1
            active[key]=dict(job=job['id'],process=launch(job,script,args,'HPO-query-'+key))
        save(WORK/'status.json',dict(state='training' if features else 'waiting_or_extracting_tokens',
            completed=len(done),total=len(cases),active={k:v['job'] for k,v in active.items()},updated_at_utc=now()))
        time.sleep(10)
    save(WORK/'status.json',dict(state='summarizing',completed=len(cases),total=len(cases),updated_at_utc=now()))
    restore_background()
    with (WORK/'logs/summary.private.log').open('ab') as f:
        subprocess.run([str(PYTHON),str(WORK/'summarize.py')],env=environment(),stdout=f,stderr=f,check=True)
    deadline=time.monotonic()+600
    while load(WORK/'results.json').get('state')!='complete':
        if time.monotonic()>deadline:raise RuntimeError('summary_receipt_not_visible')
        time.sleep(10)
    save(WORK/'status.json',dict(state='complete',completed=len(cases),total=len(cases),updated_at_utc=now()))
    try:
        notify('query fusion complete',json.dumps(load(WORK/'results.json'),indent=2))
        save(WORK/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:save(WORK/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        restore_background()
        try:notify('query fusion needs inspection','Local experiment controller failed: '+type(e).__name__+'. Existing allocations and frozen baseline are preserved.')
        except Exception:pass
        raise
