"""Persistent one-model supervisor: renew ended leases and restart full checkpoints."""
import contextlib
import fcntl
import os
import re
import signal
import socket
import subprocess
import time
from configuration import WORK,VENDOR,BASE,V1,DIAGNOSTICS
from common import load,save,now,environment,SLURM,PYTHON,notify
from batch_job import preflight

TERMINAL={'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY',
          'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}


def command(args,timeout=30):
    return subprocess.run([str(SLURM/args[0]),*args[1:]],capture_output=True,
                          text=True,timeout=timeout,check=True).stdout


def queue():
    result=command(['squeue','-h','-u','research_user','-o','%i|%j|%T|%N'])
    return {v[0]:dict(name=v[1],state=v[2],node=v[3])
            for line in result.splitlines() if len(v:=line.strip().split('|'))==4}


def accounting(ids):
    if not ids:return {}
    result=command(['sacct','-n','-X','-j',','.join(ids),'-o','JobIDRaw,State','-P'])
    return {v[0]:v[1].split()[0].rstrip('+') for line in result.splitlines()
            if len(v:=line.strip().split('|'))>=2 and v[0] in ids}


def replacement_name(slot,old):
    return f'cranio-ten25-s{int(slot):02d}-r{old}'


def matches_by_name(current,name):
    return [job for job,value in current.items() if value['name']==name]


def reconcile_pool(current):
    registry=load(WORK/'pool_registry.json');jobs=dict(registry['jobs'])
    if len(jobs)!=20 or len(set(jobs.values()))!=20:raise RuntimeError('twenty_unique_slots_required')
    missing=[job for job in jobs.values() if job not in current]
    ended=accounting(missing)
    journal=load(WORK/'replacement_journal.json')
    for slot,old in jobs.items():
        if ended.get(old) not in TERMINAL:continue
        name=replacement_name(slot,old)
        record=journal.get(name,{})
        found=matches_by_name(current,name)
        if not found and record.get('attempted'):
            # Acknowledgement may have been lost; look beyond the live queue.
            result=command(['sacct','-n','-X','-u','research_user','--name='+name,
                '--starttime='+record['started_at_utc'][:10],'-o','JobIDRaw,JobName%100','-P'])
            found=list({v[0] for line in result.splitlines()
                if len(v:=line.strip().split('|'))>=2 and v[1]==name and v[0].isdigit()})
        if len(found)>1:raise RuntimeError('duplicate_named_replacement_requires_reconciliation')
        if found:
            job=found[0]
            journal.setdefault(name,dict(attempted=True,started_at_utc=now(),slot=slot,old_job=old))
        elif record.get('attempted') and not record.get('explicit_rejection'):
            save(WORK/'replacement_status.json',dict(state='reconciling_ambiguous_submission',slot=slot,updated_at_utc=now()))
            continue  # Never blindly resubmit an ambiguous sbatch.
        else:
            save(WORK/'submission_in_progress.json',dict(state='replacing_terminal_lease',updated_at_utc=now()))
            journal[name]=dict(attempted=True,started_at_utc=now(),slot=slot,old_job=old)
            save(WORK/'replacement_journal.json',journal)
            try:
                job=command(['sbatch','--parsable','--job-name='+name,str(WORK/'lease.sbatch'),slot],timeout=60).strip().split(';')[0]
            except subprocess.CalledProcessError as error:
                message=(error.stderr or '').lower()
                rejected=any(x in message for x in ['job violates accounting/qos policy','invalid account','invalid partition','qosmaxsubmitjobperuserlimit','job submit limit'])
                journal[name]['explicit_rejection']=rejected
                save(WORK/'replacement_journal.json',journal)
                continue
            except subprocess.TimeoutExpired:
                continue
            if not job.isdigit():continue
        registry['jobs'][slot]=job
        save(WORK/'pool_registry.json',registry)
        journal[name].update(accepted_job=job,reconciled_at_utc=now())
        save(WORK/'replacement_journal.json',journal)
        from pool_work import release_owner_after_exit
        release_owner_after_exit(old)
        (WORK/'submission_in_progress.json').unlink(missing_ok=True)


def cleanup_steps():
    ids=set(load(WORK/'pool_registry.json')['jobs'].values())
    ids.update(v['job_id'] for v in load(WORK/'active_pool.json').get('allocations',[]))
    result=command(['squeue','--steps','-h','-u','research_user','-o','%i|%j'])
    steps=[]
    for line in result.splitlines():
        values=line.strip().split('|')
        if len(values)==2:
            step,name=values
            if re.fullmatch(r'\d+\.\d+',step) and step.split('.')[0] in ids and name.startswith('cranio-pool-'):
                steps.append(step)
    if steps:command(['scancel',*steps])
    return not steps


def start_identity(pid):
    try:
        with open(f'/proc/{pid}/stat') as stream:fields=stream.read().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,IndexError):return None


def same_process(record):
    return bool(record.get('host',socket.gethostname())==socket.gethostname() and
                record.get('pid') and record.get('start_identity') and
                start_identity(record['pid'])==record['start_identity'])


def stop_group(record,force=False):
    if not record.get('pid'):return
    if record.get('host',socket.gethostname())!=socket.gethostname():return
    identity=start_identity(record['pid'])
    if identity is not None and identity!=record.get('start_identity'):return
    # Only groups created and persisted by this supervisor are addressed.
    with contextlib.suppress(ProcessLookupError):os.killpg(record['pid'],signal.SIGKILL if force else signal.SIGTERM)


def launch_controller():
    for name in ['controller_exit.json','controller_heartbeat.json']:(WORK/name).unlink(missing_ok=True)
    env=environment();env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
    env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (WORK/'persistent_controller.private.log').open('ab') as stream:
        proc=subprocess.Popen([str(PYTHON),str(WORK/'pool_controller.py')],env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    record=dict(pid=proc.pid,start_identity=start_identity(proc.pid),started_at=time.time(),host=socket.gethostname())
    save(WORK/'supervised_controller.json',record)
    return proc,record


def status(state,**extra):
    row=dict(state=state,updated_at_utc=now(),**extra)
    save(WORK/'supervisor_heartbeat.json',row);save(WORK/'launch_status.json',row)


def main():
    os.umask(0o077);preflight()
    proc=None;record={};restarting=False;cooldown=0.;failures=0;termination_started=None
    with (WORK/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=load(WORK/'supervised_controller.json')
        # Adopt the existing controller after a supervisor restart.
        if not same_process(record):record={}
        while True:
            status('persistent_supervisor_polling',controller_pid=record.get('pid'))
            try:
                current=queue()
                jobs=load(WORK/'pool_registry.json')['jobs']
                done=load(WORK/'pool_done.json')
                if done.get('completion_policy')=='ten_active_training_days_and_loss_plateau':
                    if same_process(record):time.sleep(10);continue
                    if not cleanup_steps():time.sleep(5);continue
                    live=[j for j in jobs.values() if j in current]
                    if live:command(['scancel',*live]);time.sleep(10);continue
                    status('ten_training_days_and_loss_plateau_complete');return
                if (WORK/'user_stop_requested.json').exists():
                    stop_group(record)
                    live=[j for j in jobs.values() if j in current]
                    if live:command(['scancel',*live])
                    status('explicit_user_stop');return
                reconcile_pool(current)
                jobs=load(WORK/'pool_registry.json')['jobs']
                ready=[j for j in jobs.values() if current.get(j,{}).get('state')=='RUNNING'
                    and load(WORK/'leases'/f'{j}.json').get('job_id')==j]
                pending=sum(current.get(j,{}).get('state') in ('PENDING','CONFIGURING') for j in jobs.values())
                alive=proc.poll() is None if proc is not None else same_process(record)
                if alive:
                    hb=WORK/'controller_heartbeat.json'
                    recent=hb.stat().st_mtime if hb.exists() else record['started_at']
                    if time.time()-recent>600:
                        termination_started=termination_started or time.monotonic()
                        stop_group(record,force=time.monotonic()-termination_started>=30);restarting=True
                        status('recovering_stale_controller',checkpoint_preserved=True)
                    else:
                        status('supervising_training_controller',controller_pid=record['pid'],
                            registered_running_gpus=len(ready),pending_single_gpu_jobs=pending)
                else:
                    termination_started=None
                    if record:
                        stop_group(record);restarting=True
                        if proc is not None:proc.wait()
                        proc=None;record={};save(WORK/'supervised_controller.json',{})
                        failures+=1;cooldown=time.monotonic()+min(300,30*failures)
                    if restarting:
                        if not cleanup_steps():time.sleep(5);continue
                        request=load(WORK/'stop_requested.json')
                        job=request.get('job_id')
                        if job in jobs.values() and request.get('reason')=='lease_walltime_warning':
                            if job in current:command(['scancel',job]);time.sleep(10);continue
                        (WORK/'stop_requested.json').unlink(missing_ok=True)
                        restarting=False
                    if ready and time.monotonic()>=cooldown:
                        proc,record=launch_controller()
                        status('controller_started_or_recovered',controller_pid=record['pid'],registered_running_gpus=len(ready))
                    else:
                        status('waiting_for_allocations_or_recovery',registered_running_gpus=len(ready),pending_single_gpu_jobs=pending,
                               minimum_training_gpus=2,minimum_active_training_days=10)
                time.sleep(10)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError):
                status('scheduler_or_filesystem_retrying_training_retained');time.sleep(30)
            except Exception as error:
                status('supervisor_reconciliation_retrying',error_type=type(error).__name__)
                notify('ten day supervisor reconciliation','Craniofacial supervisor is reconciling resources',
                    'The persistent supervisor retained the current model and is retrying resource reconciliation. Checkpoints and patient data remain on HPC.')
                time.sleep(60)
