"""Start one encoder with available single-GPU jobs and expand at safe checkpoints."""
import contextlib
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS,WORLD
from common import load,save,now,environment,SLURM,PYTHON,notify
from batch_job import preflight
from incremental import choose_world,exposure,verified_boundary,snapshot,SUPPORTED_WORLDS
from pool_work import release_owner_after_exit
from checkpoint_recovery import recover,pin
from run_policy import CONTINUE_STATES,COMPLETE_STATE


def query(args):
    result=subprocess.run([str(SLURM/args[0]),*args[1:]],capture_output=True,text=True,timeout=15,check=True)
    return result.stdout


def select_resources(registry,records):
    ids=list(registry['jobs'].values())
    if len(ids)!=WORLD or len(set(ids))!=WORLD or not all(j.isdigit() for j in ids):
        raise RuntimeError('twenty_unique_single_gpu_jobs_required')
    result=[]
    for job in ids:
        d=records[job]
        if d.get('JobId')!=job or not d.get('UserId','').startswith('research_user('):raise RuntimeError('allocation_owner_or_identity_mismatch')
        if d.get('JobState') in ('PENDING','CONFIGURING'):continue
        if d.get('JobState')!='RUNNING':continue
        tres=dict(x.split('=',1) for x in d.get('AllocTRES','').split(',') if '=' in x)
        if d.get('Partition')!='gpu-xe9680q' or tres.get('gres/gpu')!='1' or int(tres.get('cpu',0))<4:
            raise RuntimeError('single_h100_allocation_required')
        memory=tres.get('mem','')
        if not memory.endswith('G') or float(memory[:-1])<96:raise RuntimeError('single_gpu_host_memory_mismatch')
        receipt=load(WORK/'leases'/f'{job}.json')
        if receipt.get('job_id')==job and receipt.get('node')==d.get('NodeList'):
            result.append(dict(job_id=job,node=d['NodeList']))
    return sorted(result,key=lambda d:(d['node'],int(d['job_id'])))


class SchedulerUnavailable(RuntimeError):pass


_VALIDATED_ALLOCATIONS={}


def resources():
    registry=load(WORK/'pool_registry.json')
    ids=list(registry['jobs'].values())
    try:
        # One queue request per poll, not twenty scontrol RPCs. Fully validate
        # each allocation once when it first becomes RUNNING.
        result=query(['squeue','-h','-u','research_user','-o','%i|%T|%N'])
        current={}
        for line in result.splitlines():
            job,state,node=line.strip().split('|')
            if job in ids:current[job]=(state,node)
        records={}
        for job in ids:
            state,node=current.get(job,('UNKNOWN',''))
            if state in ('PENDING','CONFIGURING'):
                records[job]=dict(JobId=job,UserId='research_user(',JobState=state)
            elif state=='RUNNING':
                d=_VALIDATED_ALLOCATIONS.get(job)
                if d is None:
                    d=dict(x.split('=',1) for x in query(['scontrol','show','job',job,'-o']).split() if '=' in x)
                    _VALIDATED_ALLOCATIONS[job]=d
                d=dict(d,JobState=state,NodeList=node)
                records[job]=d
            else:records[job]=dict(JobId=job,UserId='research_user(',JobState=state)
        return select_resources(registry,records)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
        raise SchedulerUnavailable('transient_slurm_query_failure') from error


def heartbeat(state,**extra):
    save(WORK/'controller_heartbeat.json',dict(state=state,updated_at_utc=now(),**extra))


def step_command(allocation,name,script,*args):
    return [str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--job-name=cranio-pool-'+name,
        '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1',
        '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/script),*args]


def worker_environment():
    env=environment()
    env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
    env.update(OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return env


def stop_clients(clients):
    for proc in clients:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):proc.terminate()
    for proc in clients:
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):proc.kill()
            proc.wait(timeout=20)


class ExtraDiagnostics:
    """After the legacy lease workers yield, use nontraining allocations productively."""
    def __init__(self):self.workers={};self.finished=set();self.counter=0
    def update(self,available,training):
        used={d['job_id'] for d in training}
        for d in available:
            job=d['job_id']
            if job in used or job in self.finished:continue
            if load(WORK/'leases'/f'{job}.json').get('state')!='available':continue
            old=self.workers.get(job)
            if old:
                if old['process'].poll() is not None:
                    old['process'].wait();release_owner_after_exit(job)
                    self.finished.add(job);self.workers.pop(job)
                continue
            release_owner_after_exit(job)  # Legacy parent confirms its diagnostic child exited.
            self.counter+=1;stop=WORK/'extra_diagnostics'/f'{job}_{self.counter}.stop.json'
            env=worker_environment();env.update(POOL_PREWORK_IGNORE_GLOBAL_JOIN='1',STAGING_STOP_FILE=str(stop),STAGING_OWNER=job)
            log=WORK/'extra_diagnostics'/f'{job}_{self.counter}.private.log';log.parent.mkdir(exist_ok=True)
            with log.open('ab') as stream:
                proc=subprocess.Popen(step_command(d,'diagnostic','staging_worker.py'),env=env,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=stream)
            self.workers[job]=dict(process=proc,stop=stop)
    def stop(self,jobs=None):
        selected=[(j,v) for j,v in self.workers.items() if jobs is None or j in jobs]
        for job,value in selected:save(value['stop'],dict(reason='join_at_checkpoint',updated_at_utc=now()))
        deadline=time.monotonic()+300
        while any(v['process'].poll() is None for _,v in selected) and time.monotonic()<deadline:
            heartbeat('yielding_extra_diagnostics');time.sleep(5)
        stop_clients([v['process'] for _,v in selected])
        for job,value in selected:
            release_owner_after_exit(job);self.workers.pop(job,None);self.finished.discard(job)


def run_step(pool,mode,port,plan_path,extra):
    plan=load(plan_path);world=plan['world_size'];arm=plan['arm']
    folder=WORK/arm/('gpu_probe' if mode=='probe' else 'training')
    folder.mkdir(parents=True,exist_ok=True)
    clients=[];last_activity=time.monotonic();marker=None;last_query=0
    try:
        for rank,allocation in enumerate(pool):
            env=worker_environment()
            env.update(POOL_ATTEMPT_PLAN=str(plan_path),POOL_RANK=str(rank),COMPARISON_PORT=str(port),MASTER_PORT=str(port),
                MASTER_ADDR=pool[0]['node'],COMPARISON_MASTER=pool[0]['node'])
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(step_command(allocation,mode,'pool_worker.py',mode),env=env,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=stream))
        while True:
            codes=[p.poll() for p in clients]
            if any(code not in (None,0) for code in codes):raise RuntimeError('distributed_pool_worker_failed')
            if all(code==0 for code in codes):break
            paths=list(folder.glob('rank_*.json'))+list(folder.glob('initialization_rank_*.json'))
            paths+=list(folder.glob('resume_rank_*.json'))+list(folder.glob('validation/step_*/report.json'))
            paths+=list(folder.glob('ckpt/*/.metadata'))
            if (folder/'progress.json').exists():paths.append(folder/'progress.json')
            newest=max((p.stat().st_mtime for p in paths),default=None)
            if newest!=marker:marker=newest;last_activity=time.monotonic()
            if time.monotonic()-last_activity>(12*60 if mode=='probe' else 30*60):raise RuntimeError('no_verified_pool_progress_timeout')
            if time.monotonic()-last_query>=20:
                try:available=resources()
                except SchedulerUnavailable:
                    heartbeat('scheduler_temporarily_unavailable_training_continues',world_size=world)
                    last_query=time.monotonic();time.sleep(5);continue
                last_query=time.monotonic()
                if {d['job_id'] for d in pool}-{d['job_id'] for d in available}:
                    raise RuntimeError('active_training_lease_lost_recover_checkpoint')
                extra.update(available,pool)
                # All candidate world sizes divide every 25-step sample increment,
                # so the upcoming checkpoint offset remains exactly representable.
                target=choose_world(len(available),plan['consumed_global_samples'],current=world)
                if target>world:
                    save(WORK/arm/'resize_requested.json',dict(reason='new_allocations_ready',target_world=target,
                        from_world=world,requires_complete_checkpoint_and_passing_validation=True,requested_at_utc=now()))
                save(WORK/'status.json',dict(state='running',phase=mode,world_size=world,
                    allocated_registered_gpus=len(available),diagnostic_workers=len(extra.workers),
                    expansion_requested_to=target if target>world else None,arm=arm,updated_at_utc=now()))
            heartbeat('running',phase=mode,world_size=world,arm=arm)
            time.sleep(10)
    finally:stop_clients(clients)


def prepare_attempt(plan,path):
    save(path,plan)
    env=worker_environment();env['POOL_ATTEMPT_PLAN']=str(path)
    code="from configuration import *;from common import save;from omegaconf import OmegaConf;folder=WORK/ARMS[0]/'training';folder.mkdir(parents=True,exist_ok=False);OmegaConf.save(configuration(ARMS[0]),folder/'config.yaml');save(folder.parent/'protocol.json',protocol())"
    subprocess.run([str(PYTHON),'-c',code],env=env,check=True,timeout=120)


def wait_yield(pool,extra):
    extra.stop({d['job_id'] for d in pool})
    deadline=time.monotonic()+10*60
    while not all(load(WORK/'leases'/f"{d['job_id']}.json").get('state')=='available' for d in pool):
        if time.monotonic()>deadline:raise RuntimeError('diagnostic_yield_timeout')
        heartbeat('waiting_diagnostic_yield');time.sleep(5)
    for d in pool:release_owner_after_exit(d['job_id'])


def main():
    os.umask(0o077);preflight()
    signal.signal(signal.SIGUSR1,lambda *_:save(WORK/'stop_requested.json',dict(reason='coordinator_walltime_warning',updated_at_utc=now())))
    with (WORK/'pool_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        heartbeat('starting_or_recovering')
        (WORK/'attempts').mkdir(exist_ok=True)
        extra=ExtraDiagnostics();minimum=load(WORK/'capacity_policy.json').get('minimum_world',2)
        start_plan,attempt=recover(WORK)
        source=Path(start_plan['source_training']);source_world=start_plan['source_world'];completed=start_plan['completed_steps'];consumed=start_plan['consumed_global_samples']
        credited=start_plan.get('training_seconds_at_resume',0.)
        prior=[]
        try:
            while True:
                while True:
                    try:available=resources()
                    except SchedulerUnavailable:
                        heartbeat('scheduler_temporarily_unavailable_retrying')
                        save(WORK/'status.json',dict(state='scheduler_temporarily_unavailable_retrying',checkpoint_preserved=True,updated_at_utc=now()))
                        time.sleep(20);continue
                    world=choose_world(len(available),consumed,minimum=minimum)
                    if world:break
                    extra.update(available,[])
                    save(WORK/'status.json',dict(state='waiting_for_minimum_training_pool',allocated_registered_gpus=len(available),
                        minimum_training_gpus=minimum,target_gpus=20,completed_steps=completed,updated_at_utc=now()))
                    heartbeat('waiting_for_minimum_training_pool')
                    time.sleep(20)
                # Preserve existing rank placement where possible, then append new cards.
                active_ids={d['job_id'] for d in available}
                selected=[d for d in prior if d['job_id'] in active_ids]
                selected += [d for d in available if d['job_id'] not in {x['job_id'] for x in selected}]
                pool=selected[:world]
                save(WORK/'join_requested.json',dict(state='legacy_workers_yield_for_incremental_pool',updated_at_utc=now()))
                wait_yield(pool,extra)
                extra.update(available,pool)
                arm=f'cranio_stage_{attempt:03d}_w{world:02d}'
                plan=dict(arm=arm,world_size=world,completed_steps=completed,source_training=str(source),
                    source_world=source_world,consumed_global_samples=consumed,batch_size_per_gpu=32,target_steps=completed+5000,
                    training_seconds_at_resume=credited,
                    checkpoint_period=25,independent_models=1,optimizer_reset=False,learning_rate_rescaled=False,
                    created_at_utc=now())
                path=WORK/'attempts'/f'{attempt:03d}.json';prepare_attempt(plan,path)
                save(WORK/'active_pool.json',dict(allocations=pool,world_size=world,attempt_plan=str(path),selected_at_utc=now()))
                port=30000+(int(pool[0]['job_id'])+attempt*17)%19000
                run_step(pool,'probe',port,path,extra)
                proofs=[load(WORK/arm/'gpu_probe'/f'rank_{r:02d}.json') for r in range(world)]
                if not all(r.get('state')=='passed' and r.get('optimizer_reshard_next_update_matches') for r in proofs):
                    raise RuntimeError('pool_reshard_probe_failed')
                notify(f'incremental stage {attempt} started','Craniofacial pretraining stage starting',
                    f'One shared model is restoring a full checkpoint on {world} H100 GPUs. Extra allocations run local diagnostics until a validated checkpoint permits expansion. Private artifacts remain on HPC.')
                try:
                    run_step(pool,'train',port,path,extra)
                except RuntimeError:
                    # Only a confirmed capacity failure permits a larger-world retry
                    # from the unchanged source. Other failures require inspection.
                    logs=list((WORK/arm/'training').glob('rank_*.private.log'))
                    capacity=any(any(token in p.read_text(errors='replace') for token in
                        ('torch.OutOfMemoryError','CUDA out of memory','host_memory_headroom_guard','oom-kill event')) for p in logs)
                    if capacity and world<20 and not list((WORK/arm/'training/ckpt').glob('*/.metadata')):
                        save(WORK/arm/'capacity_retry.json',dict(state='insufficient_capacity',source_checkpoint_preserved=True,
                            world_size=world,completed_steps=completed,created_at_utc=now()))
                        minimum=min(n for n in SUPPORTED_WORLDS if n>world)
                        save(WORK/'capacity_policy.json',dict(minimum_world=minimum,updated_at_utc=now()))
                        attempt+=1;prior=pool
                        continue
                    raise
                folder=WORK/arm/'training';row=verified_boundary(folder,world)
                expected=exposure(consumed,completed,row['completed_steps'],world)
                if row['consumed_global_samples']!=expected:raise RuntimeError('sample_exposure_receipt_mismatch')
                save(WORK/'status.json',dict(state=row['state'],completed_steps=row['completed_steps'],
                    world_size=world,consumed_global_samples=expected,arm=arm,updated_at_utc=now()))
                if row['state']==COMPLETE_STATE:
                    save(WORK/'pool_done.json',dict(state='finished',completion_policy='ten_active_training_days_and_loss_plateau',updated_at_utc=now()))
                    notify('ten day training complete','Craniofacial pretraining completed ten active days and reached a loss plateau',
                        'The shared model completed the minimum active training duration and the operational loss plateau criterion. Checkpoints and aggregate validation reports remain on HPC.')
                    return
                if row['state'] not in CONTINUE_STATES:
                    raise RuntimeError('training_health_guard_requires_checkpoint_recovery')
                boundary=load(folder/'checkpoint_ready'/f'step_{row["completed_steps"]:06d}.json')
                cursor=pin(WORK,folder,boundary,attempt)
                completed=cursor['completed_steps'];consumed=expected;credited=cursor['training_seconds_at_resume']
                source=Path(cursor['source_training']);source_world=world;prior=pool;attempt+=1
                save(WORK/'latest_resize_boundary.json',dict(state='verified',completed_steps=completed,source_world=world,
                    consumed_global_samples=consumed,source_training=str(source),updated_at_utc=now()))
                if row['state']=='paused_walltime_checkpoint_preserved':
                    save(WORK/'controller_exit.json',dict(state='checkpointed_lease_rotation',updated_at_utc=now()))
                    return
        finally:extra.stop()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(WORK/'controller_exit.json',dict(state='retryable_failure',error_type=type(error).__name__,updated_at_utc=now()))
        save(WORK/'status.json',dict(state='recovering_from_checkpoint',error_type=type(error).__name__,updated_at_utc=now()))
        notify('ten day automatic checkpoint recovery','Craniofacial training is recovering from a checkpoint',
            'The ten-day supervisor is restarting the shared workflow from a validated full checkpoint. Existing usable GPU allocations are retained. No private data is included.')
        raise
