"""Checkpoint-based DDP resizing across existing single-H100 Slurm leases."""
import contextlib
import fcntl
import os
import subprocess
import time
from control import WORK, OLD, PYTHON, SLURM, load, save, now, environment, digest

def command(args,timeout=30):
    return subprocess.run([str(SLURM/args[0]),*args[1:]],env=environment(),capture_output=True,
        text=True,timeout=timeout,check=True).stdout

def resources():
    jobs=set(load(OLD/'pool_registry.json')['jobs'].values())
    text=command(['squeue','-h','-u','research_user','-o','%i|%T|%N'])
    ready=[]
    for line in text.splitlines():
        values=line.split('|')
        if len(values)!=3:continue
        job,state,node=values
        if job not in jobs or state!='RUNNING':continue
        receipt=load(OLD/'leases'/f'{job}.json')
        if receipt.get('state')!='available' or receipt.get('node')!=node:continue
        info=dict(v.split('=',1) for v in command(['scontrol','show','job',job,'-o']).split() if '=' in v)
        tres=dict(v.split('=',1) for v in info.get('AllocTRES','').split(',') if '=' in v)
        if not info.get('UserId','').startswith('research_user(') or info.get('Partition')!='gpu-xe9680q' or tres.get('gres/gpu')!='1':
            raise RuntimeError('unexpected_allocation_owner_or_resources')
        ready.append(dict(job_id=job,node=node))
    return sorted(ready,key=lambda r:(r['node'],int(r['job_id'])))

def cleanup_clients(clients):
    for child in clients:
        if child.poll() is None:
            child.terminate()
    for child in clients:
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill();child.wait(timeout=20)

def cleanup_stale_steps():
    jobs=set(load(OLD/'pool_registry.json')['jobs'].values())
    text=command(['squeue','--steps','-h','-u','research_user','-o','%i|%j'])
    steps=[]
    for line in text.splitlines():
        values=line.split('|')
        if len(values)==2:
            sid,name=values
            if sid.split('.')[0] in jobs and name.startswith('transface-cpt-') and sid.rsplit('.',1)[-1].isdigit():
                steps.append(sid)
    # Cancel only stale numerical steps of this encoder; never whole jobs.
    if steps:command(['scancel',*steps])
    return not steps

def verify_ready():
    cache=load(WORK/'cache_receipt.json')
    if cache.get('state')!='complete' or digest(WORK/'cache/index.private.json')!=cache.get('index_sha256'):
        raise RuntimeError('verified_input_cache_required')
    if load(WORK/'conversion_receipt.json').get('state')!='passed':
        raise RuntimeError('verified_pretrained_weights_required')
    if load(WORK/'capacity_receipt.json').get('state')!='passed':
        raise RuntimeError('passed_gpu_capacity_required')
    for name,sha in load(WORK/'code_receipt.json').items():
        if digest(WORK/name)!=sha:
            raise RuntimeError('pinned_experiment_code_modified')

def main():
    os.umask(0o077)
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_ready()
        while not cleanup_stale_steps():time.sleep(5)
        attempt=load(WORK/'attempt_counter.json').get('next_attempt',0)
        failures=0
        while not (WORK/'health_stop.json').exists() and not (WORK/'stop_requested.json').exists():
            try:pool=resources()
            except (subprocess.SubprocessError,OSError):
                time.sleep(15);continue
            if not pool:
                save(WORK/'status.json',dict(state='waiting_for_h100_allocation',updated_at_utc=now()))
                time.sleep(15);continue
            if attempt:
                while not cleanup_stale_steps():time.sleep(5)
            capacity=load(WORK/'capacity_receipt.json')
            batch=load(WORK/'batch_override.json').get('per_gpu_batch',capacity['selected_per_gpu_batch'])
            save(WORK/'attempt_counter.json',dict(next_attempt=attempt+1))
            port=33000+(attempt%1000)
            plan=dict(attempt=attempt,world_size=len(pool),allocations=pool,per_gpu_batch=batch,
                global_batch=batch*len(pool),checkpoint_at_start=load(WORK/'latest_checkpoint.json').get('step',0),
                master_addr=pool[0]['node'],master_port=port,created_at_utc=now())
            save(WORK/'attempts'/f'{attempt:04d}.json',plan);save(WORK/'active_pool.json',plan)
            clients=[];last_progress=time.monotonic();marker=None;start=time.monotonic();last_query=0.
            try:
                for rank,allocation in enumerate(pool):
                    env=environment();env.update(RANK=str(rank),WORLD_SIZE=str(len(pool)),LOCAL_RANK='0',
                        LOCAL_WORLD_SIZE='1',MASTER_ADDR=pool[0]['node'],MASTER_PORT=str(port),
                        ARC_ATTEMPT=str(attempt),ARC_BATCH=str(batch))
                    args=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],f'--job-name=transface-cpt-{attempt}-{rank}',
                        '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1',
                        '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
                        str(PYTHON),str(WORK/'train.py')]
                    with (WORK/'logs'/f'attempt_{attempt:04d}_rank_{rank:02d}.private.log').open('ab') as stream:
                        clients.append(subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream))
                while True:
                    codes=[c.poll() for c in clients]
                    if any(c not in (None,0) for c in codes):raise RuntimeError('arcface_worker_exited')
                    if all(c==0 for c in codes):break
                    paths=[WORK/name for name in ('status.json','latest_checkpoint.json','latest_validation.json')]
                    mt=max((p.stat().st_mtime for p in paths if p.exists()),default=0.)
                    if mt!=marker:marker=mt;last_progress=time.monotonic()
                    if time.monotonic()-last_progress>30*60:raise RuntimeError('arcface_progress_stale')
                    if time.monotonic()-last_query>30:
                        try:
                            available=resources()
                        except (subprocess.SubprocessError,OSError):
                            last_query=time.monotonic();time.sleep(10);continue
                        current={r['job_id'] for r in available};active={r['job_id'] for r in pool}
                        if active-current:raise RuntimeError('active_h100_lease_lost')
                        if current-active:
                            save(WORK/'resize_requested.json',dict(attempt=attempt,target_world=len(available),
                                reason='new_h100_allocations_available',updated_at_utc=now()))
                        save(WORK/'controller_heartbeat.json',dict(state='running',world_size=len(pool),
                            allocated_gpus=len(available),attempt=attempt,updated_at_utc=now()))
                        last_query=time.monotonic()
                    time.sleep(5)
                failures=0
            except (RuntimeError,subprocess.SubprocessError,OSError) as error:
                failures+=1
                rows=[load(WORK/'failures'/f'attempt_{attempt}_rank_{r}.json') for r in range(len(pool))]
                if any(r.get('oom') for r in rows):
                    batch=max(16,(batch//2//16)*16)
                    save(WORK/'batch_override.json',dict(per_gpu_batch=batch,reason='oom_recovery',updated_at_utc=now()))
                save(WORK/'recovery_status.json',dict(state='recovering_full_checkpoint',error_type=type(error).__name__,
                    attempt=attempt,consecutive_failures=failures,updated_at_utc=now()))
                if failures>=3:
                    save(WORK/'health_stop.json',dict(reason='three_consecutive_runtime_failures',attempt=attempt,updated_at_utc=now()))
            finally:
                cleanup_clients(clients)
            attempt+=1
            if failures:time.sleep(15)
        save(WORK/'controller_exit.json',dict(state='checkpoint_preserved_allocations_retained',updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'health_stop.json',dict(reason='controller_preflight_or_integrity_failure',
            error_type=type(error).__name__,updated_at_utc=now()))
        raise
