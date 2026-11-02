"""Persistent nine-run orchestration; preserve every parent GPU allocation."""
import fcntl
import subprocess
import time
from common import *


def launch(job,script,args,name):
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1',
         '--cpus-per-task=4','--mem=48G','--gpus-per-task=1','--overlap','--exact','--immediate=30','--mpi=none',str(PYTHON),str(WORK/script),*args]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:
        return subprocess.Popen(cmd,env=environment(),stdout=f,stderr=f)


def restore_background():
    children=[]
    for item in load(WORK/'priority_yield.json').get('queues',[]):
        qflag=Path(item['queue_flag'])
        if load(qflag).get('owner')==str(WORK):qflag.unlink()
        restored=False
        for name in item['arm_flags']:
            aflag=Path(name)
            if load(aflag).get('owner')==str(WORK):aflag.unlink();restored=True
        if restored:
            status=Path(item['round'])/'status.json'
            if load(status).get('state')=='complete':save(status,dict(state='resume_requested_after_primary',updated_at_utc=now()))
        # If the old queue is still alive, it will see the cleared flag itself.
        lock=Path(item['queue'])/'continuous_seeds.lock'
        try:
            with lock.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:continue
        with (WORK/'logs'/('restored_'+item['kind']+'.private.log')).open('ab') as f:
            children.append(subprocess.Popen([str(PYTHON),item['script']],env=environment(),stdout=f,stderr=f))
    save(WORK/'background_restore.json',dict(state='prior_queues_resume_enabled',allocations_preserved=True,updated_at_utc=now()))
    return children


def summarize(cases):
    import numpy as np
    records={name:load(WORK/'cases'/name/'result.json') for name in cases}
    arms={}
    for arm in load(WORK/'protocol.json')['arms']:
        available=[r for r in records.values() if r.get('arm')==arm and r.get('state')=='complete']
        values={}
        for task in ['diagnosis','native_visual_retrieval']:
            values[task]={f'{metric}_{stat}':float(fun([r['metrics'][task][metric] for r in available])) if available else None
                          for metric in ['top1','top5'] for stat,fun in [('mean',np.mean),('SD',lambda x:np.std(x,ddof=1) if len(x)>1 else 0.)]}
        arms[arm]=dict(completed_seeds=len(available),metrics=values,
                       selected_epochs={str(r['seed']):r['selected_epoch'] for r in available})
    result=dict(state='complete' if all(r.get('state')=='complete' for r in records.values()) else 'partial_requires_inspection',
                arms=arms,per_run=records,baseline_replaced=False,test_used=False,updated_at_utc=now())
    save(WORK/'results.json',result)
    return result


def main():
    lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    assert load(WORK/'tests.json')['state']=='passed' and load(WORK/'cache_receipt.json')['state']=='passed'
    p=load(WORK/'protocol.json');cases={f'{a}__seed{s}':(a,str(s)) for s in p['seeds'] for a in p['arms']}
    active={};attempts={};retired={};waiting={};failed=set()
    while True:
        for key,record in list(active.items()):
            code=record['process'].poll()
            if code is not None:
                del active[key];retired[key]=time.monotonic()
                if code==0:waiting[key]=time.monotonic()
                elif attempts[key]>=3:failed.add(key)
        smoke=load(WORK/'smoke.json').get('state')=='passed'
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'}
        for key,t in list(waiting.items()):
            if (key=='smoke' and smoke) or key in done:del waiting[key]
            elif time.monotonic()-t>600:failed.add(key);del waiting[key]
        if 'smoke' in failed:raise RuntimeError('GPU_smoke_failed')
        if len(done|failed)>=len(cases):break
        tasks=[('smoke','smoke.py',[])] if not smoke else [(n,'train.py',list(args)) for n,args in cases.items() if n not in done|failed]
        try:jobs=free_jobs()
        except Exception:jobs=[]
        busy={v['job'] for v in active.values()};jobs=[j for j in jobs if j['id'] not in busy]
        for key,script,args in tasks:
            if not jobs or len(active)>=9:break
            if key in active or key in waiting or time.monotonic()-retired.get(key,-1e9)<90:continue
            failure=load(WORK/'smoke_failure.json') if key=='smoke' else load(WORK/'cases'/key/'failure.json')
            if failure.get('error_type') in ['FloatingPointError','AssertionError','OutOfMemoryError','RuntimeError','ValueError']:
                failed.add(key);continue
            job=jobs.pop(0);attempts[key]=attempts.get(key,0)+1
            active[key]=dict(job=job['id'],process=launch(job,script,args,'fusion-unfreeze-'+key))
        save(WORK/'status.json',dict(state='training' if smoke else 'waiting_or_GPU_smoke',completed=len(done),total=len(cases),
             active={k:v['job'] for k,v in active.items()},failed_cases=sorted(failed),updated_at_utc=now()))
        time.sleep(10)
    result=summarize(cases)
    save(WORK/'status.json',dict(state=result['state'],completed=len(done),total=len(cases),failed_cases=sorted(failed),allocations_preserved=True,updated_at_utc=now()))
    children=restore_background()
    try:
        notify('completed',json.dumps(result['arms'],indent=2)+'\nDevelopment comparison only; frozen release unchanged.')
        save(WORK/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:save(WORK/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
    # Keep replacement background coordinators inside this persistent CPU step.
    while any(child.poll() is None for child in children):time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('requires inspection','Local unfreezing controller stopped: '+type(e).__name__+'. GPU allocations and checkpoints retained.')
        except Exception:pass
        # Do not cancel or overwrite live experiment worker allocations.
        raise SystemExit(1) from None
