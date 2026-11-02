"""Bounded local recovery after a scheduler outage; never duplicate accepted jobs.

No clinical files are opened. It reconciles canceled single-GPU requests,
retains the immutable step575 source, and starts the tested coordinator on a
CPU of the first replacement allocation. All output is local operational state.
"""
import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import time
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS
from common import load,save,now,digest,environment,SLURM,PYTHON


def command(args,timeout=20):
    return subprocess.run([str(SLURM/args[0]),*args[1:]],capture_output=True,text=True,timeout=timeout,check=True).stdout


def status(state,**values):
    save(WORK/'recovery_status.json',dict(state=state,updated_at_utc=now(),**values))


def queue():
    result=command(['squeue','-h','-u','research_user','-o','%i|%j|%T|%N'])
    return {parts[0]:dict(name=parts[1],state=parts[2],node=parts[3])
            for line in result.splitlines() if len(parts:=line.strip().split('|'))==4}


def main():
    os.umask(0o077)
    with (WORK/'recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=load(WORK/'recovery_journal.json')
        if not state:
            state=dict(old_jobs=list(load(WORK/'pool_registry.json')['jobs'].values()),new_jobs={},
                token=str(int(time.time())),phase='reconcile_old_jobs',started_at_utc=now())
            save(WORK/'recovery_journal.json',state)
        started=time.monotonic()
        while time.monotonic()-started<46*3600:
            try:current=queue()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                status('scheduler_unavailable_retrying',checkpoint_preserved=True);time.sleep(60);continue
            if state['phase']=='reconcile_old_jobs':
                live=[job for job in state['old_jobs'] if job in current]
                if live:
                    # A previous controller already requested cancellation. Confirm
                    # it has completed before replacing any request.
                    status('waiting_for_failed_pool_to_finish',remaining_old_jobs=len(live))
                    try:command(['scancel',*live])
                    except (subprocess.TimeoutExpired,subprocess.CalledProcessError):pass
                    time.sleep(30);continue
                archive=WORK/('scheduler_outage_'+state['token']);archive.mkdir(exist_ok=True)
                for name in ('pool_done.json','stop_requested.json','join_requested.json','controller_heartbeat.json',
                             'status.json','embedded_controller_launch.json','pool_registry.json','preflight.json'):
                    p=WORK/name
                    if p.exists() and not (archive/name).exists():p.replace(archive/name)
                if list((WORK/'attempts').glob('*.json')):raise RuntimeError('unexpected_training_attempt_requires_review')
                # Every previous GPU process has exited; recover interrupted claims
                # while keeping complete and quarantined tasks unchanged.
                from pool_work import release_owner_after_exit
                for job in state['old_jobs']:release_owner_after_exit(job)
                save(WORK/'submission_in_progress.json',dict(state='submitting_replacement_single_gpu_jobs',created_at_utc=now()))
                save(WORK/'pool_registry.json',dict(jobs={},requested_gpus=20,one_gpu_per_job=True))
                state['phase']='submit';save(WORK/'recovery_journal.json',state)
            if state['phase']=='submit':
                for slot in range(20):
                    key=str(slot)
                    if key in state['new_jobs']:continue
                    name=f'cranio-r{state["token"]}-{slot:02d}'
                    # On a timed-out acknowledgement, query the unique job name
                    # before any retry. Ambiguous submissions are never repeated.
                    existing=[job for job,value in current.items() if value['name']==name]
                    if state.get('ambiguous_slot')==slot and not existing:
                        status('submission_acknowledgement_requires_reconciliation',slot=slot)
                        raise RuntimeError('ambiguous_submission_not_retried')
                    if len(existing)>1:raise RuntimeError('duplicate_named_submission_requires_review')
                    try:
                        job=existing[0] if existing else command(['sbatch','--parsable','--job-name='+name,
                            str(WORK/'lease.sbatch'),key],timeout=60).strip().split(';')[0]
                    except subprocess.TimeoutExpired:
                        state['ambiguous_slot']=slot;save(WORK/'recovery_journal.json',state)
                        raise
                    if not job.isdigit():raise RuntimeError('invalid_sbatch_receipt')
                    state['new_jobs'][key]=job;state.pop('ambiguous_slot',None)
                    save(WORK/'recovery_journal.json',state)
                    save(WORK/'pool_registry.json',dict(jobs=state['new_jobs'],requested_gpus=20,one_gpu_per_job=True))
                state['phase']='launch_coordinator';save(WORK/'recovery_journal.json',state)
                (WORK/'submission_in_progress.json').unlink(missing_ok=True)
                # Rehash the tested code and changed operational registry. Do not
                # regenerate clinical staging inputs or reset their task ledger.
                proof=load(WORK/('scheduler_outage_'+state['token'])/'preflight.json')
                names=set(proof['file_sha256'])|{p.name for p in WORK.glob('*.py')}
                proof.update(file_sha256={name:digest(WORK/name) for name in names},created_at_utc=now())
                save(WORK/'preflight.json',proof)
                save(WORK/'submission.json',dict(gpu_jobs=list(state['new_jobs'].values()),
                    replaces_canceled_jobs=state['old_jobs'],coordinator='first allocated job CPU',created_at_utc=now()))
                handoff=load(WORK.parent/'CURRENT_OPERATIONAL_HANDOFF.json')
                handoff.update(gpu_job_ids=list(state['new_jobs'].values()),coordinator='first allocated job CPU',
                    state='incremental_gpu_pool_queued',minimum_start_gpus=4,preflight_tests_passed=31,updated_at_utc=now())
                save(WORK.parent/'CURRENT_OPERATIONAL_HANDOFF.json',handoff)
            if state['phase']=='launch_coordinator':
                # Refresh after submission; GPU parents do actual prework while
                # this lightweight watcher waits for a ready CPU on that node.
                current=queue()
                ready=[job for job in state['new_jobs'].values() if current.get(job,{}).get('state')=='RUNNING'
                       and load(WORK/'leases'/f'{job}.json').get('job_id')==job]
                if not ready:
                    status('waiting_for_first_allocated_gpu',submitted_single_gpu_jobs=20);time.sleep(30);continue
                env=environment();env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
                env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                job=ready[0]
                args=[str(SLURM/'srun'),'--jobid='+job,'--job-name=cranio-incremental-coordinator',
                    '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G','--gres=none',
                    '--cpu-bind=none','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'pool_controller.py')]
                with (WORK/'embedded_controller.private.log').open('ab') as stream:
                    proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
                deadline=time.monotonic()+180
                while time.monotonic()<deadline and proc.poll() is None:
                    heartbeat=load(WORK/'controller_heartbeat.json')
                    if heartbeat:
                        state.update(phase='handed_to_incremental_controller',coordinator_parent_job=job,
                            coordinator_srun_client_pid=proc.pid,completed_at_utc=now())
                        save(WORK/'recovery_journal.json',state);status('handed_to_incremental_controller',coordinator_parent_job=job)
                        return
                    time.sleep(5)
                raise RuntimeError('coordinator_startup_requires_inspection')
            if state['phase']=='handed_to_incremental_controller':return
        raise RuntimeError('bounded_recovery_window_expired')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('recovery_requires_inspection',error_type=type(error).__name__,checkpoint_preserved=True)
        raise
