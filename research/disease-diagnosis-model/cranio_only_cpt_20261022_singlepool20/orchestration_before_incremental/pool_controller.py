"""Pool twenty separately scheduled H100 jobs into one global FSDP model."""
import contextlib
import fcntl
import os
import signal
import subprocess
import time
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS,WORLD,ARMS,configuration,protocol
from common import load,save,now,environment,SLURM,PYTHON,notify
from batch_job import preflight
from omegaconf import OmegaConf


def query(args):
    result=subprocess.run([str(SLURM/args[0]),*args[1:]],capture_output=True,text=True,timeout=30,check=True)
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
        if d.get('JobState')!='RUNNING':raise RuntimeError('selected_lease_ended_before_training')
        tres=dict(x.split('=',1) for x in d.get('AllocTRES','').split(',') if '=' in x)
        if d.get('Partition')!='gpu-xe9680q' or tres.get('gres/gpu')!='1' or int(tres.get('cpu',0))<4:
            raise RuntimeError('single_h100_allocation_required')
        memory=tres.get('mem','')
        if not memory.endswith('G') or float(memory[:-1])<96:raise RuntimeError('single_gpu_host_memory_mismatch')
        receipt=load(WORK/'leases'/f'{job}.json')
        if receipt.get('job_id')==job and receipt.get('node')==d.get('NodeList'):
            result.append(dict(job_id=job,node=d['NodeList']))
    return sorted(result,key=lambda d:(d['node'],int(d['job_id'])))


def stop_clients(clients):
    for proc in clients:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
    for proc in clients:
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)


def run_step(pool,mode,port):
    folder=WORK/'gpu_probe' if mode=='probe' else WORK/ARMS[0]/'training'
    folder.mkdir(parents=True,exist_ok=True)
    clients=[];last_activity=time.monotonic();marker=None
    try:
        for rank,allocation in enumerate(pool):
            env=environment()
            env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
            env.update(POOL_RANK=str(rank),COMPARISON_PORT=str(port),MASTER_PORT=str(port),
                MASTER_ADDR=pool[0]['node'],COMPARISON_MASTER=pool[0]['node'],OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            command=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--job-name=cranio-pool20-'+mode,
                '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1',
                '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
                str(PYTHON),str(WORK/'pool_worker.py'),mode]
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=stream,start_new_session=True))
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
            save(WORK/'controller_heartbeat.json',dict(state='running',phase=mode,world_size=WORLD,updated_at_utc=now()))
            time.sleep(10)
    finally:stop_clients(clients)


def main():
    os.umask(0o077);preflight()
    registry=load(WORK/'pool_registry.json')
    with (WORK/'pool_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();pool=None
        while True:
            records={job:dict(x.split('=',1) for x in query(['scontrol','show','job',job,'-o']).split() if '=' in x)
                     for job in registry['jobs'].values()}
            pool=select_resources(registry,records)
            save(WORK/'status.json',dict(state='assembling_single_gpu_pool',allocated_registered_gpus=len(pool),target_gpus=WORLD,
                independent_models=1,updated_at_utc=now()))
            save(WORK/'controller_heartbeat.json',dict(state='assembling_pool',updated_at_utc=now()))
            if len(pool)==WORLD:break
            if time.monotonic()-started>47*3600:raise RuntimeError('bounded_pool_assembly_timeout')
            time.sleep(20)
        save(WORK/'active_pool.json',dict(allocations=pool,world_size=WORLD,selected_at_utc=now()))
        save(WORK/'join_requested.json',dict(state='yield_diagnostics_for_shared_training',updated_at_utc=now()))
        deadline=time.monotonic()+10*60
        while not all(load(WORK/'leases'/f"{d['job_id']}.json").get('state')=='available' for d in pool):
            if time.monotonic()>deadline:raise RuntimeError('diagnostic_yield_timeout')
            save(WORK/'controller_heartbeat.json',dict(state='waiting_diagnostic_yield',updated_at_utc=now()))
            time.sleep(5)
        port=30000+int(os.environ['SLURM_JOB_ID'])%20000
        save(WORK/'status.json',dict(state='running',phase='distributed_16to20_reshard_probe',world_size=WORLD,updated_at_utc=now()))
        run_step(pool,'probe',port)
        receipts=[load(WORK/'gpu_probe'/f'rank_{r:02d}.json') for r in range(WORLD)]
        if not all(r.get('state')=='passed' and r.get('optimizer_reshard_next_update_matches') for r in receipts):
            raise RuntimeError('pool_reshard_probe_failed')
        folder=WORK/ARMS[0]/'training';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'ckpt').exists():raise RuntimeError('existing_training_output_requires_review')
        OmegaConf.save(configuration(ARMS[0]),folder/'config.yaml');save(WORK/'protocol.json',protocol())
        save(WORK/'status.json',dict(state='running',phase='full_state_resume_and_training',world_size=WORLD,updated_at_utc=now()))
        notify('singlepool20 started','Twenty single-GPU allocations joined for craniofacial pretraining',
            'The independent single-GPU jobs formed one distributed model. The full-state synthetic test passed; local checkpoint restoration and training follow. All private artifacts remain on HPC.')
        run_step(pool,'train',port)
        rows=[load(folder/f'rank_{r:02d}_receipt.json') for r in range(WORLD)]
        if not all(r.get('success') for r in rows) or len({(r.get('state'),r.get('completed_steps')) for r in rows})!=1:
            raise RuntimeError('inconsistent_pool_completion_receipts')
        save(WORK/'status.json',dict(state=rows[0]['state'],completed_steps=rows[0]['completed_steps'],world_size=WORLD,completed_at_utc=now()))
        notify('singlepool20 '+rows[0]['state'],'Craniofacial pooled pretraining completed or paused',
            'The shared model completed its bounded budget or stopped at a declared guard. Complete checkpoints remain on HPC. The single-GPU batch jobs now exit.')


if __name__=='__main__':
    success=False
    try:main();success=True
    except BaseException as error:
        save(WORK/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        notify('singlepool20 failure','Craniofacial single-GPU pool requires inspection',
            'A resource, integrity, runtime or progress check stopped the pooled workflow. Completed checkpoints are preserved locally.')
        raise
    finally:
        save(WORK/'pool_done.json',dict(state='finished' if success else 'failed',updated_at_utc=now()))
        registry=load(WORK/'pool_registry.json')
        if not success and registry.get('jobs'):
            subprocess.run([str(SLURM/'scancel'),*registry['jobs'].values()],capture_output=True,timeout=30)
