import os,sys,time,subprocess
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import free_jobs as query_free_jobs,environment,SLURM,notify

def free_jobs():
    while True:
        try:return query_free_jobs()
        except (OSError,subprocess.SubprocessError):time.sleep(10)

def launch(job,script,args,name):
    e=environment();e['PYTHONPATH']=str(WORK/'dependencies')+':'+str(PROJECT/'arcface_cpt_20261026/dependencies')+':'+e['PYTHONPATH']
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*map(str,args)]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:return subprocess.Popen(cmd,env=e,stdout=f,stderr=f)

def main():
    os.umask(0o077);p=load(WORK/'protocol.json');assert load(WORK/'split_audit.json').get('state')=='passed';assert load(WORK/'model_test.json').get('state')=='passed'
    cases=[(a,s,a+'_'+str(s)) for a in p['arms'] for s in p['seeds']];active={};attempts={}
    for a,s,n in cases:(WORK/n).mkdir(exist_ok=True)
    while True:
        done={n for a,s,n in cases if load(WORK/n/'result.json').get('state')=='complete'}
        if len(done)==len(cases):break
        for n,proc in list(active.items()):
            if proc.poll() is not None:
                if proc.returncode and attempts[n]>=2 and load(WORK/n/'result.json').get('state')!='complete':raise RuntimeError('repeated_R50_training_failure')
                del active[n]
        done={n for a,s,n in cases if load(WORK/n/'result.json').get('state')=='complete'}
        pool=free_jobs()
        for a,s,n in cases:
            if n in done or n in active or not pool:continue
            attempts[n]=attempts.get(n,0)+1;active[n]=launch(pool.pop(0),'train.py',[a,s],'R50-clean-'+n)
        save(WORK/'status.json',dict(state='R50_training',completed_cases=len(done),total_cases=len(cases),active_cases=list(active),updated_at_utc=now()));time.sleep(15)
    for case in ['pretrained']+[n for a,s,n in cases]:
        if load(WORK/(case+'_inference.json')).get('state')=='complete':continue
        while not free_jobs():time.sleep(15)
        proc=launch(free_jobs()[0],'evaluate.py',[case],'R50-clean-infer-'+case)
        if proc.wait():raise RuntimeError('R50_inference_failed')
    with (WORK/'logs/score.private.log').open('ab') as f:subprocess.run([str(PYTHON),str(WORK/'evaluate.py'),'score'],env=environment(),stdout=f,stderr=f,check=True)
    save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('Clean R50 reproduction complete',__import__('json').dumps(load(WORK/'results.json'),indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        notify('Clean R50 reproduction requires inspection',type(e).__name__+'. Allocations and private data retained.');raise
