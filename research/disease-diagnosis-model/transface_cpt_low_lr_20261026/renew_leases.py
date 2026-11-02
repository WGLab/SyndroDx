"""Renew only terminal leases, preserving all live job IDs and queue priority."""
import subprocess
from control import WORK, OLD, SLURM, load, save, now, environment

TERMINAL={'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}

def command(args,timeout=30):
    return subprocess.run([str(SLURM/args[0]),*args[1:]],capture_output=True,text=True,
        env=environment(),timeout=timeout,check=True).stdout

def reconcile():
    registry=load(OLD/'pool_registry.json');jobs=dict(registry['jobs'])
    live={}
    for line in command(['squeue','-h','-u','research_user','-o','%i|%j|%T']).splitlines():
        fields=line.split('|')
        if len(fields)==3:live[fields[0]]=dict(name=fields[1],state=fields[2])
    missing=[job for job in jobs.values() if job not in live]
    if not missing:return
    ended={}
    for line in command(['sacct','-n','-X','-j',','.join(missing),'-o','JobIDRaw,State','-P']).splitlines():
        fields=line.split('|')
        if len(fields)>=2:ended[fields[0]]=fields[1].split()[0].rstrip('+')
    journal=load(WORK/'replacement_journal.json')
    for slot,old in jobs.items():
        if ended.get(old) not in TERMINAL:continue
        name=f'arcface-s{int(slot):02d}-r{old}'
        found=[j for j,row in live.items() if row['name']==name]
        record=journal.get(name,{})
        if not found and record.get('attempted'):
            text=command(['sacct','-n','-X','-u','research_user','--name='+name,
                '--starttime='+record['started_at_utc'][:10],'-o','JobIDRaw,JobName%100','-P'])
            found=list({v[0] for line in text.splitlines() if len(v:=line.split('|'))>=2 and v[1]==name and v[0].isdigit()})
        if len(found)>1:raise RuntimeError('ambiguous_lease_replacement')
        if found:job=found[0]
        elif record.get('attempted') and not record.get('explicit_rejection'):
            continue
        else:
            journal[name]=dict(attempted=True,slot=slot,old_job=old,started_at_utc=now())
            save(WORK/'replacement_journal.json',journal)
            save(OLD/'submission_in_progress.json',dict(state='arcface_terminal_lease_replacement',updated_at_utc=now()))
            try:
                job=command(['sbatch','--parsable','--job-name='+name,str(OLD/'lease.sbatch'),slot],60).strip().split(';')[0]
            except subprocess.CalledProcessError as error:
                text=(error.stderr or '').lower()
                journal[name]['explicit_rejection']=any(k in text for k in ('job violates accounting/qos policy','invalid account','invalid partition','job submit limit'))
                save(WORK/'replacement_journal.json',journal)
                continue
            except subprocess.TimeoutExpired:continue
            if not job.isdigit():continue
        registry['jobs'][slot]=job;save(OLD/'pool_registry.json',registry)
        journal[name]['accepted_job']=job;save(WORK/'replacement_journal.json',journal)
        (OLD/'submission_in_progress.json').unlink(missing_ok=True)
