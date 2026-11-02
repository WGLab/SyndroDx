"""Fixed protocol, equal search budgets, all seeds, allocation-preserving retries."""
import os,sys,time,subprocess,fcntl
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import free_jobs as query_free_jobs,environment,SLURM,notify
REMOTE_STEPS=set()

def refresh_steps():
    global REMOTE_STEPS
    while True:
        try:
            REMOTE_STEPS=set(subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%j'],env=environment(),capture_output=True,text=True,timeout=40,check=True).stdout.splitlines());return
        except (subprocess.SubprocessError,OSError):time.sleep(10)

class ExistingWorker:
    def __init__(self,name,prefix):self.name=name;self.prefix=prefix
    def poll(self):return None if self.prefix+self.name in REMOTE_STEPS else 0

def free_jobs():
    while True:
        try:return query_free_jobs()
        except (subprocess.SubprocessError,OSError):time.sleep(10)

def launch(job,script,args,name):
    e=environment();e['PYTHONPATH']=str(WORK/'dependencies')+':'+str(WORK/'vendor')+':'+str(PROJECT/'arcface_cpt_20261026/dependencies')+':'+e['PYTHONPATH'];e['MPLCONFIGDIR']='/tmp/matched_benchmark_mpl'
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*args]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:return subprocess.Popen(cmd,env=e,stdout=f,stderr=f,start_new_session=True)

def main():
    os.umask(0o077);lock=(WORK/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert load(WORK/'audit.json').get('state')=='passed' and load(WORK/'tests.json').get('state')=='passed'
    assert load(WORK/'arcface_gradient_test.json').get('state')=='passed'
    assert load(WORK/'audit.json')['protocol_sha256']==digest(WORK/'protocol.json')
    p=load(WORK/'protocol.json');cases={}
    for regime in p['regimes']:
        for lr in p['learning_rates'][regime]:
            for arm in p['arms']:
                for seed in p['seeds']:
                    name=f'{arm}__{regime}__lr{lr}__seed{seed}';cases[name]=dict(arm=arm,regime=regime,lr=lr,seed=seed)
    save(WORK/'cases.json',dict(cases=cases,protocol_sha256=digest(WORK/'protocol.json')))
    order=sorted(cases,key=lambda n:(cases[n]['lr']!=p['common_recipe_lr'][cases[n]['regime']],cases[n]['regime'],cases[n]['arm']!='transface_chop_cpt',cases[n]['seed']))
    for n in cases:(WORK/'cases'/n).mkdir(parents=True,exist_ok=True)
    active={};attempts={};grace={}
    refresh_steps()
    for n in cases:
        if 'matched-train-'+n in REMOTE_STEPS:active[n]=ExistingWorker(n,'matched-train-');attempts[n]=1
    while True:
        refresh_steps()
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        for n,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if code==0 and n not in done:
                grace.setdefault(n,time.monotonic())
                if time.monotonic()-grace[n]<180:continue
            if code and attempts[n]>=3 and n not in done:raise RuntimeError('repeated_case_failure_'+n)
            del active[n]
        done={n for n in cases if load(WORK/'cases'/n/'result.json').get('state')=='complete'};pool=free_jobs()
        for n in order:
            if n in done or n in active or not pool:continue
            attempts[n]=attempts.get(n,0)+1
            active[n]=launch(pool.pop(0),'train.py',[n],'matched-train-'+n)
        save(WORK/'status.json',dict(state='matched_training',completed_cases=len(done),total_cases=len(cases),active_cases=list(active),updated_at_utc=now()));time.sleep(15)
    settings={'common_recipe':{},'equal_budget_tuned':{}};LRs={}
    for arm in p['arms']:
        for regime in p['regimes']:
            metrics={}
            for lr in p['learning_rates'][regime]:
                names=[n for n,c in cases.items() if c['arm']==arm and c['regime']==regime and c['lr']==lr]
                results=[load(WORK/'cases'/n/'result.json')['validation'] for n in names]
                metrics[lr]=(sum(r['top5'] for r in results)/len(results),sum(r['top1'] for r in results)/len(results))
            selected=max(metrics,key=lambda lr:(*metrics[lr],-lr));cell=arm+'__'+regime;LRs[cell]=dict(selected_lr=selected,validation_scores=metrics)
            for setting,lr in [('common_recipe',p['common_recipe_lr'][regime]),('equal_budget_tuned',selected)]:
                settings[setting][cell]=[n for seed in p['seeds'] for n,c in cases.items() if c==dict(arm=arm,regime=regime,lr=lr,seed=seed)]
    initializations=['init__'+arm for arm in p['arms']]
    save(WORK/'selection.json',dict(settings=settings,frozen_initializations=initializations,development_only_LR_selection=LRs,all_seeds_retained=True,updated_at_utc=now()))
    required=sorted({n for cells in settings.values() for names in cells.values() for n in names}|set(initializations));active={};attempts={};grace={}
    refresh_steps()
    for n in required:
        if 'matched-infer-'+n in REMOTE_STEPS:active[n]=ExistingWorker(n,'matched-infer-');attempts[n]=1
    while True:
        refresh_steps()
        done={n for n in required if load(WORK/'inference'/f'{n}.json').get('state')=='complete'}
        if len(done)==len(required):break
        for n,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if code==0 and n not in done:
                grace.setdefault(n,time.monotonic())
                if time.monotonic()-grace[n]<180:continue
            if code and attempts[n]>=3 and n not in done:raise RuntimeError('repeated_inference_failure')
            del active[n]
        pool=free_jobs()
        for n in required:
            if n in done or n in active or not pool:continue
            attempts[n]=attempts.get(n,0)+1;active[n]=launch(pool.pop(0),'evaluate.py',[n],'matched-infer-'+n)
        save(WORK/'status.json',dict(state='matched_test_inference',completed=len(done),total=len(required),active=len(active),updated_at_utc=now()));time.sleep(15)
    save(WORK/'gpu_work_complete.json',dict(state='complete',updated_at_utc=now()))
    save(WORK/'status.json',dict(state='paired_statistics',updated_at_utc=now()))
    with (WORK/'logs/score.private.log').open('ab') as f:subprocess.run([str(PYTHON),str(WORK/'score.py')],env=environment(),stdout=f,stderr=f,check=True)
    save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('Matched craniofacial benchmark complete',__import__('json').dumps(load(WORK/'results.json'),indent=2));save(WORK/'notification.json',dict(state='notifier_invoked',updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));notify('Matched benchmark requires inspection',type(e).__name__+'. Checkpoints and allocations retained.');raise
