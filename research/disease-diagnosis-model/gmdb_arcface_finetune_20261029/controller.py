"""Two-GPU supervised comparison while remaining cards run identity seeds."""
import sys,os,time,subprocess
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import free_jobs,environment,SLURM,notify

def launch(job,script,args,name):
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*map(str,args)]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:return subprocess.Popen(cmd,env=environment(),stdout=f,stderr=f)

def main():
    os.umask(0o077);p=load(WORK/'protocol.json');cases=[]
    for arm in p['arms']:
        for lr in p['learning_rates']:
            for seed in p['seeds']:
                name=f'{arm}_lr{lr}_seed{seed}';(WORK/name).mkdir(exist_ok=True);cases.append((arm,lr,seed,name))
    save(WORK/'cases.json',dict(cases=cases))
    paused=None
    if not free_jobs():
        q=load(PROJECT/'arcface_transfer_evaluation_20261028/seed_queue_status.json')
        if q.get('path'):
            root=Path(q['path'])
            for a in load(root/'protocol.json').get('arms',{}):
                flag=root/a/'pause_requested.json'
                if not flag.exists() and load(root/a/'exit.json').get('state')!='complete':
                    save(flag,dict(reason='temporary_checkpoint_yield_for_GMDB_disease_finetuning',updated_at_utc=now()));paused=flag;break
    active={};attempts={}
    while True:
        done={c[3] for c in cases if load(WORK/c[3]/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        for name,proc in list(active.items()):
            if proc.poll() is not None:
                if proc.returncode and attempts[name]>=2:raise RuntimeError('repeated_training_failure')
                del active[name]
        pool=free_jobs()
        for c in cases:
            name=c[3]
            if name in done or name in active or len(active)>=2 or not pool:continue
            attempts[name]=attempts.get(name,0)+1
            active[name]=launch(pool.pop(0),'train.py',c,'gmdb-disease-'+name)
        if paused and active:
            out=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%j'],env=environment(),capture_output=True,text=True,timeout=40,check=True).stdout
            if any(n.startswith('gmdb-disease-') for n in out.splitlines()):paused.unlink(missing_ok=True);paused=None
        save(WORK/'status.json',dict(state='disease_finetuning',completed_cases=len(done),total_cases=len(cases),active_cases=list(active),updated_at_utc=now()));time.sleep(15)
    selected={}
    for arm in p['arms']:
        candidates=[load(WORK/c[3]/'best.json') for c in cases if c[0]==arm]
        selected[arm]=max(candidates,key=lambda c:(c['metrics']['top5'],c['metrics']['top1']))
    save(WORK/'selected.json',selected)
    for arm in p['arms']:
        while not free_jobs():time.sleep(20)
        proc=launch(free_jobs()[0],'evaluate.py',[arm],'gmdb-finetuned-inference-'+arm)
        if proc.wait():raise RuntimeError('evaluation_inference_failed')
    proc=subprocess.run([str(PYTHON),str(WORK/'evaluate.py'),'score'],env=environment(),check=True)
    save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('GMDB disease fine-tuning complete',__import__('json').dumps(load(WORK/'results.json'),indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        notify('GMDB fine-tuning requires inspection',type(e).__name__+'. Allocations and checkpoints retained.');raise
