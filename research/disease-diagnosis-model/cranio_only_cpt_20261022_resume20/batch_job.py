"""A gang allocation runs real diagnostics and training, then releases itself."""
import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path
from configuration import WORK,BASE,V1,VENDOR,DIAGNOSTICS,WORLD,ARMS,configuration,protocol
from common import load,save,digest,now,environment,SLURM,PYTHON,notify
from omegaconf import OmegaConf


def preflight():
    proof=load(WORK/'preflight.json')
    if proof.get('state')!='passed' or proof.get('tests_passed',0)<12:
        raise RuntimeError('preflight_tests_required')
    for name,sha in proof['file_sha256'].items():
        if digest(WORK/name)!=sha:raise RuntimeError('code_or_anchor_changed_after_preflight')
    plan=load(WORK/'resume_plan.json')
    if not plan.get('source_validation_paused') or not plan.get('explicit_user_resume_authorization'):
        raise RuntimeError('explicit_post_regression_resume_authorization_required')
    path=WORK/'source_snapshot/ckpt/574'
    if digest(path/'.metadata')!=plan['metadata_sha256']:raise RuntimeError('source_metadata_modified')
    for name,size in plan['checkpoint_file_sizes'].items():
        f=path/name
        if not f.is_file() or f.stat().st_size!=size:raise RuntimeError('source_checkpoint_incomplete')
    if load(WORK/'probe_audit/status.json').get('state')!='complete':
        raise RuntimeError('cached_feature_audit_required')


def worker_step(mode,env):
    folder=WORK/'gpu_probe' if mode=='probe' else WORK/ARMS[0]/'training'
    folder.mkdir(parents=True,exist_ok=True)
    shared=['--gpus-per-task=h100:1','--cpus-per-task=4','--gpu-bind=single:1',
        '--cpu-bind=none','--mpi=none','--distribution=block:block',
        '--output='+str(folder/'slurm_rank_%t.private.log'),
        '--error='+str(folder/'slurm_rank_%t.private.log')]
    application=[str(PYTHON),str(WORK/'worker.py'),mode]
    command=[str(SLURM/'srun'),'--kill-on-bad-exit=1','--het-group=0','--nodes=2','--ntasks=16',
        *shared,*application,':','--het-group=1','--nodes=1','--ntasks=4',*shared,*application]
    proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,start_new_session=True)
    started=time.monotonic();last_activity=started;marker=None
    try:
        while proc.poll() is None:
            paths=list(folder.glob('rank_*.json'))+list(folder.glob('initialization_rank_*.json'))
            paths+=list(folder.glob('resume_rank_*.json'))+list(folder.glob('validation/step_*/report.json'))
            paths+=list(folder.glob('ckpt/*/.metadata'))
            if (folder/'progress.json').exists():paths.append(folder/'progress.json')
            newest=max((p.stat().st_mtime for p in paths),default=None)
            if newest!=marker:marker=newest;last_activity=time.monotonic()
            if time.monotonic()-last_activity>(12*60 if mode=='probe' else 30*60):
                raise RuntimeError('no_verified_progress_timeout')
            save(WORK/'heartbeat.json',dict(state='running',phase=mode,world_size=WORLD,
                job_id=os.environ['SLURM_JOB_ID'],updated_at_utc=now()))
            time.sleep(10)
        if proc.returncode:raise RuntimeError('distributed_worker_step_failed')
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)


def main():
    os.umask(0o077);preflight()
    env=environment()
    # All GPUs now belong to the same gang allocation, so ordinary NCCL peer
    # communication can be tested instead of retaining the old cross-job bypass.
    env.pop('NCCL_P2P_DISABLE',None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
    if int(os.environ.get('SLURM_HET_SIZE','0'))!=2:raise RuntimeError('expected_two_resource_components')
    count=sum(int(os.environ[f'SLURM_NTASKS_HET_GROUP_{i}']) for i in range(2))
    if count!=WORLD:raise RuntimeError('allocation_task_count_mismatch')
    nodes=[]
    for i in range(2):
        expression=os.environ[f'SLURM_JOB_NODELIST_HET_GROUP_{i}']
        nodes.extend(subprocess.run([str(SLURM/'scontrol'),'show','hostnames',expression],
            capture_output=True,text=True,timeout=30,check=True).stdout.splitlines())
    port=30000+int(os.environ['SLURM_JOB_ID'])%20000
    env.update(MASTER_ADDR=nodes[0],COMPARISON_MASTER=nodes[0],MASTER_PORT=str(port),COMPARISON_PORT=str(port))
    signal.signal(signal.SIGUSR1,lambda *_:save(WORK/'stop_requested.json',dict(reason='walltime_warning',updated_at_utc=now())))
    save(WORK/'status.json',dict(state='running',phase='synthetic_16to20_reshard_probe',job_id=os.environ['SLURM_JOB_ID'],world_size=WORLD,updated_at_utc=now()))
    worker_step('probe',env)
    receipts=[load(WORK/'gpu_probe'/f'rank_{r:02d}.json') for r in range(WORLD)]
    if not all(r.get('state')=='passed' and r.get('optimizer_reshard_next_update_matches') for r in receipts):
        raise RuntimeError('reshard_probe_proofs_incomplete')
    save(WORK/'gpu_probe/completion.json',dict(state='passed',ranks=WORLD,completed_at_utc=now()))
    folder=WORK/ARMS[0]/'training';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'ckpt').exists():raise RuntimeError('existing_output_requires_explicit_resume')
    OmegaConf.save(configuration(ARMS[0]),folder/'config.yaml')
    save(WORK/'protocol.json',protocol())
    save(WORK/'status.json',dict(state='running',phase='full_state_resume_and_training',job_id=os.environ['SLURM_JOB_ID'],world_size=WORLD,updated_at_utc=now()))
    notify('resume20 started','Craniofacial twenty-GPU pretraining allocation started',
        'A single clinical-only model is resuming from the preserved checkpoint. Synthetic distributed-state tests passed. Original-baseline comparisons and declared continuation guards remain enabled. All private artifacts stay on HPC.')
    worker_step('train',env)
    rows=[load(folder/f'rank_{r:02d}_receipt.json') for r in range(WORLD)]
    if not all(r.get('success') for r in rows) or len({(r.get('state'),r.get('completed_steps')) for r in rows})!=1:
        raise RuntimeError('inconsistent_final_rank_receipts')
    save(WORK/'status.json',dict(state=rows[0]['state'],completed_steps=rows[0]['completed_steps'],world_size=WORLD,completed_at_utc=now(),allocation_released_after_exit=True))
    notify('resume20 completed '+rows[0]['state'],'Craniofacial pretraining completed or paused',
        'The single-model run completed its bounded budget or stopped at a declared guard. Complete checkpoints and aggregate reports are on HPC. The batch allocation exits rather than retaining idle GPUs.')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(WORK/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now(),automatic_restart=False))
        notify('resume20 failure','Craniofacial pretraining requires inspection',
            'A local integrity, runtime, communication or progress check stopped the job. Completed checkpoints remain on HPC; the allocation is not held idle.')
        raise
