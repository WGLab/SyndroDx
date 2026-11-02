"""Checkpointed switch to identity ArcFace, retaining all GPU lease jobs."""
import fcntl
import os
import signal
import socket
import subprocess
import time
import datetime
from control import WORK,OLD,PYTHON,SLURM,load,save,now,environment,identity,alive,digest
PREVIOUS=WORK.parent/'transface_cpt_low_lr_20261026'

def controller_running(record):
    if record.get('host')==socket.gethostname():return alive(record)
    # A durable Slurm coordinator may adopt a controller on the VDI host.
    # The shared controller lock is still the final protection against duplication.
    heartbeat=load(WORK/'controller_heartbeat.json')
    stamp=heartbeat.get('updated_at_utc')
    if not stamp:return False
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(stamp)).total_seconds()
    return age<180

def handoff_ready(exit_record,requested):
    return (exit_record.get('state')=='checkpoint_preserved_allocations_retained'
        and bool(requested.get('requested_at_utc'))
        and exit_record.get('updated_at_utc','')>=requested['requested_at_utc'])

def heartbeat(state,**extra):
    row=dict(state=state,updated_at_utc=now(),**extra)
    save(WORK/'supervisor_status.json',row)
    save(OLD/'supervisor_heartbeat.json',dict(row,owner=WORK.name))
    status=load(WORK/'status.json');checkpoint=load(WORK/'latest_checkpoint.json')
    save(WORK.parent/'CURRENT_OPERATIONAL_HANDOFF.json',dict(updated_at_utc=now(),
        active_experiment=str(WORK),model='TransFace-L with patient-identity ArcFace objective',
        training_state=status.get('state',state),status_file=str(WORK/'status.json'),
        supervisor_record=str(WORK/'supervisor_process.json'),checkpoint_record=str(WORK/'latest_checkpoint.json'),
        validation_record=str(WORK/'latest_validation.json'),
        last_observed_step=status.get('completed_steps'),last_observed_checkpoint=checkpoint.get('step'),
        training_world_size=status.get('world_size'),
        privacy='Only aggregate metrics leave HPC; no images, identifiers or private mappings.',
        stage2='Disease-supervised weighted-cross-entropy adaptation is planned, not launched.'))

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
        if load(WORK/'data_preflight.json').get('state')!='passed':raise RuntimeError('passed_patient_split_preflight_required')
        for name,sha in load(WORK/'code_receipt.json').items():
            if digest(WORK/name)!=sha:raise RuntimeError('pinned_code_changed')
        if not (WORK/'transition.json').exists():
            save(PREVIOUS/'stop_requested.json',dict(reason='user_switch_to_patient_identity_arcface',requested_at_utc=now()))
            terminate_verified(load(PREVIOUS/'supervisor_process.json'),PREVIOUS/'supervisor.py')
            terminate_verified(load(PREVIOUS/'gpu_monitor_process.json'),PREVIOUS/'monitor_gpus.py')
        with (OLD/'launcher.lock').open('a') as pool_lock:
            fcntl.flock(pool_lock,fcntl.LOCK_EX)
            heartbeat('switching_to_identity_arcface_retaining_allocations')
            if not (WORK/'transition.json').exists():
                from controller import command
                jobs=set(load(OLD/'pool_registry.json')['jobs'].values())
                # The predecessor observes stop_requested only after a full checkpoint.
                # Keep its controller alive until all workers finish normally.
                deadline=time.monotonic()+1800
                while True:
                    steps=[]
                    for line in command(['squeue','--steps','-h','-u','research_user','-o','%i|%j']).splitlines():
                        fields=line.split('|')
                        if len(fields)!=2:continue
                        sid,name=fields
                        if sid.split('.')[0] in jobs and name.startswith('transface-low-') and sid.rsplit('.',1)[-1].isdigit():steps.append(sid)
                    if not steps:break
                    if time.monotonic()>deadline:raise RuntimeError('predecessor_checkpoint_transition_timeout')
                    heartbeat('waiting_for_predecessor_checkpoint',allocations_retained=True,
                        predecessor_status=load(PREVIOUS/'status.json').get('state'))
                    time.sleep(5)
                terminate_verified(load(PREVIOUS/'controller_process.json'),PREVIOUS/'controller.py')
                previous_checkpoint=load(PREVIOUS/'latest_checkpoint.json')
                if previous_checkpoint.get('state')!='complete':raise RuntimeError('predecessor_checkpoint_required')
                save(PREVIOUS/'status.json',dict(state='checkpointed_switch_to_identity_arcface',
                    completed_steps=previous_checkpoint['step'],updated_at_utc=now()))
                save(WORK/'transition.json',dict(state='previous_transface_run_checkpointed_allocations_retained',
                    previous_status=load(PREVIOUS/'status.json'),preserved_checkpoint=load(PREVIOUS/'latest_checkpoint.json'),
                    previous_degraded_checkpoint_not_used_for_initialization=True,cancelled_steps=[],
                    cancelled_allocation_jobs=[],updated_at_utc=now()))
            proc=None;adopted=load(WORK/'controller_process.json');probe=None
            probe_adopted=load(WORK/'capacity_process.json');last_reconcile=0.
            while True:
                if (WORK/'handoff_requested.json').exists() and (WORK/'stop_requested.json').exists():
                    exit_record=load(WORK/'controller_exit.json')
                    requested=load(WORK/'stop_requested.json')
                    if handoff_ready(exit_record,requested):
                        heartbeat('retired_for_follow_on',allocations_retained=True)
                        save(WORK/'supervisor_retired.json',dict(state='complete',updated_at_utc=now()))
                        return
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
                            args=[str(SLURM/'srun'),'--jobid='+pool[0]['job_id'],'--job-name=arcface-id-capacity',
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
                    running=proc.poll() is None if proc is not None else controller_running(adopted)
                    if not running:
                        with (WORK/'logs/controller.private.log').open('ab') as out:
                            proc=subprocess.Popen([str(PYTHON),str(WORK/'controller.py')],env=environment(),
                                stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
                        adopted=dict(pid=proc.pid,start_identity=identity(proc.pid),host=socket.gethostname(),updated_at_utc=now())
                        save(WORK/'controller_process.json',adopted)
                    heartbeat('supervising_transface_training',controller_pid=adopted.get('pid'),allocations_retained=True)
                time.sleep(10)

if __name__=='__main__':main()
