"""Persistent allocation-aware comparison; cancels only its own worker steps on failure."""
import os,sys,time,subprocess,fcntl
from matched_runtime import *

def launch(arm,jobs,attempt,micro):
    folder=ROOT/arm;folder.joinpath('resize_requested.json').unlink(missing_ok=True)
    port=42000+ARMS.index(arm)*500+attempt%400;clients=[]
    save(folder/'active_pool.json',dict(attempt=attempt,jobs=jobs,world=len(jobs),updated_at_utc=now()))
    for rank,j in enumerate(jobs):
        e=environment();e.update(RANK=str(rank),WORLD_SIZE=str(len(jobs)),LOCAL_RANK='0',MASTER_ADDR=jobs[0]['node'],MASTER_PORT=str(port),MATCH_ATTEMPT=str(attempt),MATCH_MICRO=str(micro))
        cmd=[str(SLURM/'srun'),'--jobid='+j['id'],'--job-name=objective-'+arm+'-'+str(attempt)+'-'+str(rank),
            '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(ROOT/'train.py'),arm]
        with (folder/'logs'/f'attempt_{attempt}_rank_{rank}.private.log').open('ab') as log:
            clients.append(subprocess.Popen(cmd,env=e,stdout=log,stderr=log,start_new_session=True))
    return dict(clients=clients,jobs=jobs,attempt=attempt,last_progress=time.monotonic(),marker=None,started=time.monotonic())

def stop_workers(arm,run):
    # Only numerical steps with this exact experiment prefix, never allocations.
    prefix='objective-'+arm+'-'+str(run['attempt'])+'-'
    steps=command(['squeue','--steps','-u','research_user','-h','-o','%i|%j'])
    ids=[line.split('|')[0] for line in steps.splitlines() if '|' in line and line.split('|')[1].startswith(prefix)]
    if ids:
        try:command(['scancel',*ids])
        except subprocess.CalledProcessError:
            # A failing step may disappear between squeue and scancel.
            remaining=command(['squeue','--steps','-u','research_user','-h','-o','%i'])
            if set(ids)&set(remaining.splitlines()):raise
    for child in run['clients']:
        if child.poll() is None:child.terminate()

class AdoptedWorker:
    def __init__(self,name,folder,attempt):self.name=name;self.folder=folder;self.attempt=attempt
    def poll(self):
        names=command(['squeue','--steps','-u','research_user','-h','-o','%j']).splitlines()
        if self.name in names:return None
        return 0 if load(self.folder/'exit.json').get('attempt')==self.attempt else 1
    def terminate(self):pass

def main():
    os.umask(0o077);lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(ROOT/'controller_process.json',dict(pid=os.getpid(),host=__import__('socket').gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),updated_at_utc=now()))
    assert load(ROOT/'audit.json').get('state')=='passed'
    p=load(ROOT/'protocol.json');active={};scoring={};attempts={a:load(ROOT/a/'attempt.json').get('next',0) for a in ARMS};micros={a:960 for a in ARMS};failures={a:0 for a in ARMS}
    # Reattach live workers after a controller restart; do not duplicate or stop them.
    names=command(['squeue','--steps','-u','research_user','-h','-o','%j']).splitlines()
    for arm in ARMS:
        plan=load(ROOT/arm/'active_pool.json')
        if not plan:continue
        worker_names=['objective-'+arm+'-'+str(plan['attempt'])+'-'+str(i) for i in range(plan['world'])]
        if any(n in names for n in worker_names):
            active[arm]=dict(clients=[AdoptedWorker(n,ROOT/arm,plan['attempt']) for n in worker_names],jobs=plan['jobs'],attempt=plan['attempt'],last_progress=time.monotonic(),marker=None,started=time.monotonic())
    save(ROOT.parent/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(ROOT),workflow_job=os.environ.get('SLURM_JOB_ID'),status_file=str(ROOT/'status.json'),results_file=str(ROOT/'results.json'),updated_at_utc=now()))
    while True:
        try:pool=free_jobs()
        except Exception:time.sleep(15);continue
        owned={j['id'] for run in active.values() for j in run['jobs']};pool=[j for j in pool if j['id'] not in owned]
        for arm,run in list(active.items()):
            codes=[c.poll() for c in run['clients']];folder=ROOT/arm
            status=load(folder/'status.json');marker=(status.get('updated_at_utc'),load(folder/'latest_checkpoint.json').get('updated_at_utc'))
            if marker!=run['marker']:run['marker']=marker;run['last_progress']=time.monotonic()
            failed=any(c not in (None,0) for c in codes) or time.monotonic()-run['last_progress']>1800
            if failed:
                receipts=[load(folder/'failures'/f"attempt_{run['attempt']}_rank_{r}.json") for r in range(len(run['jobs']))]
                stop_workers(arm,run);failures[arm]+=1
                if any(x.get('oom') for x in receipts):micros[arm]=max(64,micros[arm]//2)
                numerical=any(x.get('error_type')=='FloatingPointError' for x in receipts)
                save(folder/'recovery.json',dict(state='inspection_required' if numerical else 'recovering',attempt=run['attempt'],failures=failures[arm],updated_at_utc=now()))
                if numerical or failures[arm]>=3:
                    save(folder/'pause_requested.json',dict(reason='numerical_or_repeated_runtime_failure',updated_at_utc=now()))
                    notify(arm+' requires inspection','A comparison arm failed its safety check. Its checkpoint and GPU allocations are retained. No patient information is included.')
                del active[arm]
            elif all(c==0 for c in codes):
                exitrec=load(folder/'exit.json')
                if exitrec.get('attempt')==run['attempt']:del active[arm]
            elif pool and len(active)==len(ARMS) and len(run['jobs'])<5 and p['global_patient_draws_per_step']%(len(run['jobs'])+1)==0:
                # Request a clean checkpoint-boundary restart. New GPU isn't launched mid-step.
                if not (folder/'resize_requested.json').exists():save(folder/'resize_requested.json',dict(reason='new_free_gpu',updated_at_utc=now()))
        waiting=[a for a in ARMS if a not in active and not (ROOT/a/'pause_requested.json').exists() and load(ROOT/a/'exit.json').get('state')!='complete']
        for pos,arm in enumerate(waiting):
            if not pool:break
            count=min(5,max(1,len(pool)//(len(waiting)-pos)))
            while p['global_patient_draws_per_step']%count:count-=1
            jobs=pool[:count];pool=pool[count:];attempt=attempts[arm];attempts[arm]+=1
            save(ROOT/arm/'attempt.json',dict(next=attempts[arm]));active[arm]=launch(arm,jobs,attempt,micros[arm])
        for folder,proc in list(scoring.items()):
            if proc.poll() is not None:del scoring[folder]
        for arm in ARMS:
            folder=ROOT/arm
            for ev in sorted((folder/'evaluations').glob('step_*')):
                if len(scoring)>=2:break
                if not (ev/'ready.json').exists() or (ev/'results.json').exists() or ev in scoring or (ev/'score_failure.json').exists():continue
                # Baseline must finish first for paired intervals.
                if ev.name!='step_0000000' and not (folder/'evaluations/step_0000000/results.json').exists():continue
                with (ROOT/'logs'/f'{arm}_{ev.name}_score.private.log').open('ab') as log:
                    scoring[ev]=subprocess.Popen([str(PYTHON),str(ROOT/'score.py'),str(ev)],env=environment(),stdout=log,stderr=log)
            reports=[(ev,load(ev/'results.json')) for ev in sorted((folder/'evaluations').glob('step_*')) if (ev/'results.json').exists()]
            if reports:
                latest=reports[-1][1];save(folder/'latest_validation.json',latest)
                adverse=[r for _,r in reports if r.get('step',0)>=500]
                decline=len(adverse)>=2 and all(r.get('gmdb_paired_to_original',{}).get('top1',{}).get('paired_patient_bootstrap_95_interval',[0,0])[1]<-.02 for r in adverse[-2:])
                if (latest.get('severe_collapse') or decline) and not (folder/'pause_requested.json').exists():
                    save(folder/'pause_requested.json',dict(reason='severe_collapse' if latest.get('severe_collapse') else 'two_supported_development_declines_over_2pp',step=latest['step'],updated_at_utc=now()))
                    notify(arm+' development guard','A prespecified representation guard requested checkpointed inspection. Other comparison arms continue. GPU allocations are preserved.')
        states={a:dict(training=load(ROOT/a/'status.json'),exit=load(ROOT/a/'exit.json'),validation_step=load(ROOT/a/'latest_validation.json').get('step'),paused=(ROOT/a/'pause_requested.json').exists()) for a in ARMS}
        save(ROOT/'status.json',dict(state='comparison_running',active_gpus=sum(len(v['jobs']) for v in active.values()),arms=states,scoring_jobs=len(scoring),updated_at_utc=now()))
        finished=all(load(ROOT/a/'exit.json').get('state')=='complete' or (ROOT/a/'pause_requested.json').exists() for a in ARMS)
        pending_scores=any((ev/'ready.json').exists() and not (ev/'results.json').exists() and not (ev/'score_failure.json').exists() for a in ARMS for ev in (ROOT/a/'evaluations').glob('step_*'))
        if finished and not active and not scoring and not pending_scores:break
        time.sleep(10)
    results={a:[load(ev/'results.json') for ev in sorted((ROOT/a/'evaluations').glob('step_*')) if (ev/'results.json').exists()] for a in ARMS}
    failure_count=sum((ev/'score_failure.json').exists() for a in ARMS for ev in (ROOT/a/'evaluations').glob('step_*'))
    save(ROOT/'results.json',dict(state='complete' if not failure_count else 'evaluation_failures_require_inspection',arms=results,protocol=p,updated_at_utc=now()))
    save(ROOT/'status.json',dict(state='complete' if not failure_count else 'inspection_required',allocations_preserved=True,updated_at_utc=now()))
    notify('stage completed',__import__('json').dumps(dict(arms={a:load(ROOT/a/'latest_validation.json') for a in ARMS},evaluation_failures=failure_count),indent=2))
    save(ROOT/'notification.json',dict(state='notifier_invoked',updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('controller failure','Comparison controller requires inspection: '+type(e).__name__+'. Allocations and checkpoints retained.')
        except Exception:pass
        raise
