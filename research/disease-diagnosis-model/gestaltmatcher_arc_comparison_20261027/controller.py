"""Persistent matched comparison; train available branches while R50 awaits weights."""
import os,sys,subprocess,time,fcntl,importlib.util
from common import *
spec=importlib.util.spec_from_file_location('alignment_runtime',PROJECT/'transface_biomedbert_alignment_20261027/common.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
SLURM=runtime.SLURM
def jobs():
    out=subprocess.run([str(SLURM/'squeue'),'-h','-u','research_user','-o','%i|%j|%T|%N'],env=runtime.environment(),capture_output=True,text=True,check=True).stdout
    return [dict(id=v[0],node=v[3]) for s in out.splitlines() if len(v:=s.split('|'))==4 and v[1].startswith('h100-hold20-') and v[2]=='RUNNING']
def launch(job,script,arg,name):
    env=runtime.environment();env['PYTHONPATH']=str(WORK)+':'+str(PROJECT/'arcface_cpt_20261026/dependencies')
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),arg]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as log:return subprocess.Popen(cmd,env=env,stdout=log,stderr=log,start_new_session=True)
def main():
    lock=(WORK/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[(a,'train_case.py',WORK/'cases'/a/'result.json') for a in ['r100','transface','r50']]+[(a,'infer.py',WORK/'evaluation'/a/'receipt.json') for a in ['unadapted','r100','transface','r50']]
    active={};failures={}
    while True:
        pool=jobs();done=[]
        for key,(child,job) in list(active.items()):
            if child.poll() is not None:
                if child.returncode:failures[key]=failures.get(key,0)+1
                del active[key]
        used={j for c,j in active.values()};free=[j for j in pool if j['id'] not in used]
        for arm,script,result in tasks:
            key=script+':'+arm
            if load(result).get('state')=='complete':done.append(key);continue
            if key in active or failures.get(key,0)>=2:continue
            if arm=='r50' and not (WORK/'assets/glint360k_r50.onnx').is_file():continue
            if script=='infer.py' and arm!='unadapted' and load(WORK/'cases'/arm/'result.json').get('state')!='complete':continue
            if not free:continue
            job=free.pop(0);active[key]=(launch(job,script,arm,'gm-compare-'+script.split('.')[0]+'-'+arm+'-'+str(int(time.time()))),job['id'])
        save(WORK/'status.json',dict(state='running',completed=done,active=list(active),available_gpus=len(pool),
            awaiting_R50_weights=not (WORK/'assets/glint360k_r50.onnx').is_file(),failed_tasks=failures,updated_at_utc=now()))
        if all(load(WORK/'evaluation'/a/'receipt.json').get('state')=='complete' for a in ['unadapted','r100','transface']) and not load(WORK/'results.json'):
            with (WORK/'logs/scoring.private.log').open('ab') as log:subprocess.run([str(PYTHON),str(WORK/'score.py')],env=runtime.environment(),stdout=log,stderr=log,check=True)
        if len(done)==len(tasks):break
        if not active and any(v>=2 for v in failures.values()):raise RuntimeError('repeated_comparison_worker_failure')
        time.sleep(30)
    with (WORK/'logs/scoring.private.log').open('ab') as log:subprocess.run([str(PYTHON),str(WORK/'score.py')],env=runtime.environment(),stdout=log,stderr=log,check=True)
    runtime.notify('matched_Arc_comparison_complete','Matched GMDB image-only comparison finished; aggregate results at '+str(WORK/'results.json')+'. Clinical holdouts were not used.')
if __name__=='__main__':
    try:main()
    except Exception as e:save(WORK/'controller_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
