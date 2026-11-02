"""Switch R100 to TransFace-L while preserving the existing Slurm allocations."""
import fcntl
import os
import signal
import socket
import subprocess
import time
from control import WORK,OLD,PYTHON,SLURM,load,save,now,environment,identity,alive,digest
PREVIOUS=WORK.parent/'arcface_cpt_20261026'

def heartbeat(state,**extra):
    row=dict(state=state,updated_at_utc=now(),**extra)
    save(WORK/'supervisor_status.json',row)
    save(OLD/'supervisor_heartbeat.json',dict(row,owner=WORK.name))

def terminate_verified(record,script):
    pid=record.get('pid')
    if not pid or not identity(pid):return
    if record.get('host')!=socket.gethostname():raise RuntimeError('predecessor_host_mismatch')
    if record.get('start_identity') and identity(pid)!=record['start_identity']:
        raise RuntimeError('predecessor_process_reused')
    path=f'/proc/{pid}'
    if os.stat(path).st_uid!=os.getuid() or str(script).encode() not in open(path+'/cmdline','rb').read().split(b'\0'):
        raise RuntimeError('predecessor_process_identity_mismatch')
    os.kill(pid,signal.SIGTERM)

def main():
    os.umask(0o077)
    with (WORK/'supervisor.lock').open('a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if load(WORK/'conversion_receipt.json').get('state')!='passed':raise RuntimeError('verified_weights_required')
        if load(WORK/'tests_receipt.json').get('state')!='passed':raise RuntimeError('passed_synthetic_tests_required')
        for name,sha in load(WORK/'code_receipt.json').items():
            if digest(WORK/name)!=sha:raise RuntimeError('pinned_code_changed')
        if not (WORK/'transition.json').exists():
            save(PREVIOUS/'stop_requested.json',dict(reason='user_switch_to_transface_l',requested_at_utc=now()))
            terminate_verified(load(PREVIOUS/'supervisor_process.json'),PREVIOUS/'supervisor.py')
        with (OLD/'launcher.lock').open('a') as pool_lock:
            fcntl.flock(pool_lock,fcntl.LOCK_EX)
            heartbeat('switching_to_transface_retaining_allocations')
            if not (WORK/'transition.json').exists():
                terminate_verified(load(PREVIOUS/'controller_process.json'),PREVIOUS/'controller.py')
                from controller import command
                jobs=set(load(OLD/'pool_registry.json')['jobs'].values())
                steps=[]
                for line in command(['squeue','--steps','-h','-u','research_user','-o','%i|%j']).splitlines():
                    fields=line.split('|')
                    if len(fields)!=2:continue
                    sid,name=fields
                    if sid.split('.')[0] in jobs and name.startswith('arcface-cpt-') and sid.rsplit('.',1)[-1].isdigit():steps.append(sid)
                if steps:command(['scancel',*steps])
                for _ in range(60):
                    live={line.split('|')[0] for line in command(['squeue','--steps','-h','-u','research_user','-o','%i|%j']).splitlines()}
                    if not live.intersection(steps):break
                    heartbeat('waiting_for_r100_workers_to_exit',allocations_retained=True)
                    time.sleep(2)
                else:raise RuntimeError('previous_training_steps_did_not_exit')
                save(WORK/'transition.json',dict(state='r100_workers_stopped_allocations_retained',
                    previous_status=load(PREVIOUS/'status.json'),preserved_checkpoint=load(PREVIOUS/'latest_checkpoint.json'),
                    uncheckpointed_predecessor_head_warmup_not_resumed=True,cancelled_steps=steps,
                    cancelled_allocation_jobs=[],updated_at_utc=now()))
            proc=None;adopted=load(WORK/'controller_process.json');probe=None
            probe_adopted=load(WORK/'capacity_process.json');last_reconcile=0.
            while True:
                if time.monotonic()-last_reconcile>60:
                    try:
                        from renew_leases import reconcile
                        reconcile()
                    except Exception as error:
                        save(WORK/'lease_reconcile_status.json',dict(state='retrying',error_type=type(error).__name__,updated_at_utc=now()))
                    last_reconcile=time.monotonic()
                if any((WORK/n).exists() for n in ('health_stop.json','stop_requested.json','training_complete.json')):
                    heartbeat('checkpoint_preserved_review_required',allocations_retained=True);time.sleep(10);continue
                if load(WORK/'capacity_receipt.json').get('state')!='passed':
                    running=probe.poll() is None if probe is not None else alive(probe_adopted)
                    if not running:
                        if probe is not None and probe.returncode!=0:
                            save(WORK/'health_stop.json',dict(reason='gpu_capacity_preflight_failed',updated_at_utc=now()));continue
                        from controller import resources
                        try:pool=resources()
                        except (subprocess.SubprocessError,OSError):
                            heartbeat('waiting_for_scheduler_response',allocations_retained=True);time.sleep(10);continue
                        if pool:
                            args=[str(SLURM/'srun'),'--jobid='+pool[0]['job_id'],'--job-name=transface-capacity',
                                '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1',
                                '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'capacity_gpu.py')]
                            with (WORK/'logs/capacity.private.log').open('ab') as out:
                                probe=subprocess.Popen(args,env=environment(),stdin=subprocess.DEVNULL,stdout=out,stderr=out)
                            probe_adopted=dict(pid=probe.pid,start_identity=identity(probe.pid),host=socket.gethostname(),updated_at_utc=now())
                            save(WORK/'capacity_process.json',probe_adopted)
                    heartbeat('transface_gpu_capacity_preflight',allocations_retained=True)
                else:
                    # The probe may have written its receipt just before CUDA process exit.
                    if (probe is not None and probe.poll() is None) or (probe is None and alive(probe_adopted)):
                        heartbeat('finishing_gpu_capacity_preflight',allocations_retained=True);time.sleep(5);continue
                    running=proc.poll() is None if proc is not None else alive(adopted)
                    if not running:
                        with (WORK/'logs/controller.private.log').open('ab') as out:
                            proc=subprocess.Popen([str(PYTHON),str(WORK/'controller.py')],env=environment(),
                                stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
                        adopted=dict(pid=proc.pid,start_identity=identity(proc.pid),host=socket.gethostname(),updated_at_utc=now())
                        save(WORK/'controller_process.json',adopted)
                    heartbeat('supervising_transface_training',controller_pid=adopted.get('pid'),allocations_retained=True)
                time.sleep(10)

if __name__=='__main__':main()
