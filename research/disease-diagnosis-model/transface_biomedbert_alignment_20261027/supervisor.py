"""Run useful work on all granted pool GPUs; resize only after complete checkpoints."""
import fcntl
import json
import os
import subprocess
import sys
import time
import socket
from common import *

def query():
    out=subprocess.run([str(SLURM/'squeue'),'-u','research_user','-h','-o','%i|%j|%T|%N'],env=environment(),text=True,capture_output=True,check=True,timeout=30).stdout
    jobs=[]
    for line in out.splitlines():
        v=line.split('|')
        if len(v)==4 and v[1].startswith('h100-hold20-') and v[2]=='RUNNING':jobs.append(dict(job_id=v[0],node=v[3]))
    # A pool allocation may be used by another task; never overlap another workload.
    steps=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%i|%j'],env=environment(),text=True,capture_output=True,check=True,timeout=30).stdout
    busy=set()
    for line in steps.splitlines():
        sid,name=line.split('|',1)
        if sid.rsplit('.',1)[-1].isdigit() and not name.startswith('align-'):busy.add(sid.split('.')[0])
    return sorted([j for j in jobs if j['job_id'] not in busy],key=lambda j:int(j['job_id']))

def launch(job,script,args,name,extra=None):
    env=environment();env.update(extra or {})
    cmd=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--job-name='+name,'--nodes=1','--ntasks=1',
        '--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--gpu-bind=single:1','--cpu-bind=none',
        '--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*map(str,args)]
    with (WORK/'logs'/(name+'.private.log')).open('ab',buffering=0) as log:
        return subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)

def status(state,**kw):
    value=dict(state=state,**kw,updated_at_utc=now())
    save(WORK/'supervisor_status.json',value)
    save(PROJECT/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(WORK),training_state=state,
        status_file=str(WORK/'status.json'),supervisor_status_file=str(WORK/'supervisor_status.json'),
        workflow_job=os.environ.get('SLURM_JOB_ID'),privacy='Only aggregate metrics leave HPC.',updated_at_utc=now()))

def stop_clients(clients):
    # Terminate only this supervisor's srun steps. Preserve every allocation.
    import signal
    for child in clients:
        if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
    for child in clients:
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)

def main():
    os.umask(0o077)
    lock=(WORK/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(WORK/'supervisor_process.json',dict(pid=os.getpid(),host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),updated_at_utc=now()))
    while True:
        jobs=query()
        if jobs:break
        status('waiting_for_allocations');time.sleep(20)
    if load(WORK/'data_receipt.json').get('state')!='complete':raise RuntimeError('paired_patient_release_required')
    if load(WORK/'tests_receipt.json').get('state')!='passed':raise RuntimeError('alignment_tests_required')
    if load(WORK/'feature_receipt.json').get('state')!='complete':
        plan=load(WORK/'feature_plan.json')
        if not plan:
            plan=dict(world_size=len(jobs),jobs=jobs,created_at_utc=now());save(WORK/'feature_plan.json',plan)
        world=plan['world_size'];clients={};attempt=0;finished_at={}
        while True:
            done=[i for i in range(world) if load(WORK/'features'/f'shard_{i}.status.json').get('state')=='complete']
            if len(done)==world:break
            jobs=query();used={j for i,(c,j) in clients.items() if c.poll() is None};free=[j for j in jobs if j['job_id'] not in used]
            for i in range(world):
                if i in done or (i in clients and clients[i][0].poll() is None):continue
                if i in clients and clients[i][0].poll()==0:
                    # Shared filesystem metadata can lag successful compute-node writes.
                    finished_at.setdefault(i,time.monotonic())
                    if time.monotonic()-finished_at[i]<180:continue
                if not free:break
                attempt+=1;job=free.pop(0)
                if attempt>world*3:raise RuntimeError('repeated_feature_extraction_failure')
                clients[i]=(launch(job,'extract.py',[i,world],f'align-features-{i}-{attempt}'),job['job_id'])
            status('extracting_frozen_TransFace_features',allocated_gpus=len(jobs),active_shards=sum(c.poll() is None for c,j in clients.values()),complete_shards=len(done),total_shards=world)
            time.sleep(10)
        for child,job in clients.values():child.wait(timeout=90)
        status('tokenizing_and_sealing_HPO_inputs')
        with (WORK/'logs/finalize.private.log').open('ab') as log:
            subprocess.run([str(PYTHON),str(WORK/'finalize_features.py'),str(world)],env=environment(),stdout=log,stderr=log,check=True)
        try:notify('data_ready','Patient-isolated GMDB craniofacial-HPO pairs and frozen TransFace features are ready. CHOP/BCH-tagged records are excluded. BCH audit deferred by user. All records remain on HPC.')
        except Exception:save(WORK/'notification_pending.json',dict(event='data_ready',updated_at_utc=now()))
    attempts=load(WORK/'attempt_counter.json').get('attempt',0);explicit_failures=0
    while not load(WORK/'training_complete.json'):
        jobs=query()
        if not jobs:status('waiting_for_allocations');time.sleep(20);continue
        attempts+=1;save(WORK/'attempt_counter.json',dict(attempt=attempts))
        save(WORK/'resize_requested.json',dict(world_size=len(jobs),updated_at_utc=now()))
        master=jobs[0]['node'];port=str(31000+(attempts%1000))
        clients=[]
        batch=load(WORK/'batch_override.json').get('per_gpu_batch',96)
        for rank,job in enumerate(jobs):
            env=dict(RANK=str(rank),WORLD_SIZE=str(len(jobs)),LOCAL_RANK='0',MASTER_ADDR=master,MASTER_PORT=port,
                ALIGN_ATTEMPT=str(attempts),ALIGN_BATCH=str(batch))
            clients.append(launch(job,'train.py',[],f'align-train-a{attempts}-r{rank}',env))
        save(WORK/'active_training.json',dict(attempt=attempts,jobs=jobs,world_size=len(jobs),started_at_utc=now()))
        started=time.monotonic()
        while any(c.poll() is None for c in clients):
            if any(c.poll() not in (None,0) for c in clients):stop_clients(clients);break
            current=query()
            if {j['job_id'] for j in jobs}!={j['job_id'] for j in current}:
                save(WORK/'resize_requested.json',dict(world_size=len(current),updated_at_utc=now()))
            progress=load(WORK/'status.json')
            status('alignment_running',allocated_gpus=len(current),training_gpus=len(jobs),
                completed_steps=progress.get('completed_steps',0),epoch_fraction=progress.get('epoch_fraction',0),
                awaiting_checkpoint_resize=len(current)!=len(jobs))
            if time.monotonic()-started>900:
                stamp=progress.get('updated_at_utc','')
                if stamp and (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(stamp)).total_seconds()>900:
                    stop_clients(clients);raise RuntimeError('training_progress_timeout')
            time.sleep(15)
        failures=[load(WORK/'attempts'/f'{attempts}_rank_{r}.failure.json') for r in range(len(jobs))]
        if any(failures):
            explicit_failures+=1
            if any(f.get('oom') for f in failures):
                save(WORK/'batch_override.json',dict(per_gpu_batch=max(8,batch//2),reason='OOM_recovery'))
            if explicit_failures>=3:raise RuntimeError('repeated_training_failure_requires_inspection')
        time.sleep(3)
    result=load(WORK/'training_complete.json');save(WORK/'status.json',result)
    status('alignment_complete',best_checkpoint=load(WORK/'best_checkpoint.json'),allocations_preserved=True)
    notify('complete','The first TransFace-BiomedBERT alignment run completed. The selected development checkpoint and aggregate metrics are saved at '+str(WORK)+'. Clinical CHOP/BCH test cohorts were not evaluated. GPU allocations are preserved.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'supervisor_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('failure','Alignment controller requires inspection. Error type: '+type(e).__name__+'. Private diagnostic logs remain on HPC. Allocations have not been cancelled.')
        except Exception:pass
        raise
