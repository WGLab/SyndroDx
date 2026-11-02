"""Reuse independent authorized allocations; do not cancel any allocation."""
import fcntl,os,signal,socket,subprocess,time
from runtime import *
def jobs():
    lines=subprocess.run([str(SLURM/'squeue'),'-u','research_user','-h','-o','%i|%j|%T|%N'],env=env(),capture_output=True,text=True,check=True).stdout.splitlines()
    pool=[dict(id=v[0],node=v[3]) for line in lines if len(v:=line.split('|'))==4 and v[1].startswith('h100-hold20-') and v[2]=='RUNNING']
    steps=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%i|%j'],env=env(),capture_output=True,text=True,check=True).stdout.splitlines()
    busy={v[0].split('.')[0] for line in steps if len(v:=line.split('|'))==2 and v[0].rsplit('.',1)[-1].isdigit() and not v[1].startswith('cranio-qwen-')}
    return sorted([j for j in pool if j['id'] not in busy],key=lambda j:int(j['id']))
def launch(job,script,args,name,extra=None):
    e=env();e.update(extra or {})
    cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*map(str,args)]
    with (WORK/'logs'/(name+'.private.log')).open('ab') as f:return subprocess.Popen(cmd,env=e,stdout=f,stderr=f,start_new_session=True)
def status(state,**kw):
    save(WORK/'controller_status.json',dict(state=state,**kw,updated_at_utc=now()))
def wait_receipt(path,seconds=180):
    start=time.monotonic()
    while not Path(path).is_file() and time.monotonic()-start<seconds:time.sleep(5)
def stop(clients):
    for c in clients:
        if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
    for c in clients:
        try:c.wait(timeout=25)
        except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL)
def main():
    os.umask(0o077);lock=(WORK/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(WORK/'tests_receipt.json').get('state')!='passed':raise RuntimeError('synthetic_tests_required')
    if load(WORK/'data_receipt.json').get('state')!='complete':raise RuntimeError('data_release_required')
    save(PROJECT/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(WORK),workflow_job=os.environ.get('SLURM_JOB_ID'),status_file=str(WORK/'status.json'),controller_status_file=str(WORK/'controller_status.json'),updated_at_utc=now()))
    while not jobs():status('waiting_for_GPU');time.sleep(20)
    if not load(WORK/'feature_receipt.json'):
        plan=load(WORK/'feature_plan.json') or dict(world=len(jobs()))
        save(WORK/'feature_plan.json',plan);world=plan['world'];active={};tries={};finished={}
        while True:
            done={i for i in range(world) if load(WORK/'features'/f'{i}.json').get('state')=='complete'}
            if len(done)==world:break
            pool=jobs();used={j for c,j in active.values() if c.poll() is None};free=[j for j in pool if j['id'] not in used]
            for i in range(world):
                if i in done or (i in active and active[i][0].poll() is None):continue
                if i in active and active[i][0].returncode==0:
                    finished.setdefault(i,time.monotonic())
                    if time.monotonic()-finished[i]<180:continue
                if not free:break
                tries[i]=tries.get(i,0)+1
                if tries[i]>2:raise RuntimeError('repeated_feature_failure')
                job=free.pop(0);active[i]=(launch(job,'features.py',[i,world],f'cranio-qwen-feat-{i}-{tries[i]}'),job['id'])
            status('extracting_native_face_tokens',shards_done=len(done),shards=world,allocated_gpus=len(pool));time.sleep(10)
        for c,j in active.values():c.wait(timeout=120)
        with (WORK/'logs/finalize.private.log').open('ab') as f:subprocess.run([str(PYTHON),str(WORK/'features.py'),'finalize',str(world)],env=env(),stdout=f,stderr=f,check=True)
    attempt=load(WORK/'attempt.json').get('attempt',0);failures=0
    while not load(WORK/'training_complete.json'):
        pool=jobs()
        if not pool:status('waiting_for_GPU');time.sleep(20);continue
        attempt+=1;save(WORK/'attempt.json',dict(attempt=attempt));save(WORK/'resize_requested.json',dict(world_size=len(pool)))
        batch=load(WORK/'batch_override.json').get('batch',load(WORK/'protocol.json')['per_gpu_batch'])
        clients=[launch(j,'train.py',[],f'cranio-qwen-train-a{attempt}-r{rank}',dict(RANK=str(rank),WORLD_SIZE=str(len(pool)),LOCAL_RANK='0',MASTER_ADDR=pool[0]['node'],MASTER_PORT=str(32500+attempt),CRANIO_ATTEMPT=str(attempt),CRANIO_BATCH=str(batch))) for rank,j in enumerate(pool)]
        save(WORK/'active_training.json',dict(attempt=attempt,jobs=pool,world_size=len(pool),per_gpu_batch=batch,started_at_utc=now()))
        while any(c.poll() is None for c in clients):
            if any(c.poll() not in (None,0) for c in clients):stop(clients);break
            available=jobs();save(WORK/'resize_requested.json',dict(world_size=len(available)))
            status('training_workers_active',world_size=len(pool),available_gpus=len(available),attempt=attempt,progress=load(WORK/'status.json'))
            time.sleep(20)
        if all(c.returncode==0 for c in clients):
            # Successful checkpoint-resize exits are not completion. Avoid holding
            # every GPU idle for a completion receipt that will never be written.
            resizing=load(WORK/'resize_requested.json').get('world_size',len(pool))!=len(pool)
            wait_receipt(WORK/('latest_checkpoint.json' if resizing else 'training_complete.json'),180)
        errors=[load(WORK/'workers'/f'{attempt}_{r}.failure.json') for r in range(len(pool))]
        if any(errors) or any(c.returncode for c in clients):
            failures+=1
            if any(x.get('oom') for x in errors) and batch>1:save(WORK/'batch_override.json',dict(batch=max(1,batch//2),reason='measured_OOM_recovery'))
            elif failures>=2:raise RuntimeError('repeated_worker_failure')
        if failures>=3:raise RuntimeError('repeated_worker_failure')
    result=load(WORK/'training_complete.json');status('complete',result=result,allocations_preserved=True)
    notify('alignment_complete','Frozen TransFace–Qwen 32B connector alignment completed. Aggregate results and the visual-conditioning gate are saved locally at '+str(WORK/'training_complete.json')+'. GPU allocations are preserved; no clinical records are included.')
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error_type=type(e).__name__)
        try:notify('controller_failed','Local Qwen connector controller needs inspection. Error type: '+type(e).__name__+'. All patient data and private logs remain on HPC; allocations were preserved.')
        except Exception:pass
        raise
