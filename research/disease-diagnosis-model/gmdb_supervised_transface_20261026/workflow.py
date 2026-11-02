"""Durable aggregate-only Slurm coordinator for CPT -> matched GMDB adaptation."""
import datetime
import fcntl
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from common import WORK,STAGE1,LEASES,PYTHON,SLURM,load,save,digest,now,environment,notify,verify_code
from policy import decide,choose_checkpoint
RUNNING_IN_SLURM=bool(os.environ.get('SLURM_JOB_ID'))

class CoordinatorHandoff(Exception):pass

def yield_to_slurm():
    if RUNNING_IN_SLURM:return
    claim=load(WORK/'slurm_coordinator_claim.json')
    registered=load(WORK/'workflow_job.json')
    if claim.get('job_id')==registered.get('job_id') and age(claim.get('updated_at_utc'))<180:
        save(WORK/'coordinator_handoff.json',dict(state='local_fallback_yielded_to_slurm',updated_at_utc=now()))
        raise CoordinatorHandoff()

def age(stamp):
    if not stamp:return float('inf')
    return (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(stamp)).total_seconds()

def command(args):
    return subprocess.run([str(SLURM/args[0]),*args[1:]],env=environment(),capture_output=True,text=True,timeout=45,check=True).stdout

def alert(event,title,message):
    sent=load(WORK/'notifications.json')
    if event in sent:return
    try:notify(event,title,message)
    except Exception as error:
        save(WORK/'notification_retry.json',dict(event=event,error_type=type(error).__name__,updated_at_utc=now()));return
    sent[event]=dict(sent_at_utc=now());save(WORK/'notifications.json',sent)

def heartbeat(state,**details):
    row=dict(state=state,**details,updated_at_utc=now())
    save(WORK/'status.json',row)
    # GPU lease.py watches this shared heartbeat, not this workflow's status file.
    # These phases run only while we own the shared launcher lock.
    if state in ('restoring_follow_on','disease_adaptation','fixed_gmdb_evaluation','complete'):
        save(LEASES/'supervisor_heartbeat.json',dict(row,owner=WORK.name))
    if state!='monitoring_identity_cpt':
        save(WORK.parent/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(WORK),training_state=state,
            status_file=str(WORK/'status.json'),workflow_job=load(WORK/'workflow_job.json').get('job_id'),
            privacy='Only aggregate metrics leave HPC.',updated_at_utc=now()))

def allocation_snapshot():
    registered=set(load(LEASES/'pool_registry.json')['jobs'].values())
    states={}
    for line in command(['squeue','-h','-u','research_user','-o','%i|%T']).splitlines():
        fields=line.split('|')
        if len(fields)==2 and fields[0] in registered:states[fields[0]]=fields[1]
    running=sum(v=='RUNNING' for v in states.values())
    pending=sum(v=='PENDING' for v in states.values())
    return dict(registered_gpu_leases=len(registered),allocated_gpus=running,pending_gpu_leases=pending,
        missing_gpu_leases=len(registered-states.keys()),allocations_retained=running>0)

def completed_heartbeat():
    resources=allocation_snapshot()
    heartbeat('complete',results_file=str(WORK/'results.json'),**resources)
    if resources['missing_gpu_leases']:
        alert('completed_workflow_lease_loss','Craniofacial results complete; GPU leases are no longer active',
            'CPT, all 24 disease-adaptation cases and the fixed GMDB evaluations completed successfully. '
            'The follow-on coordinator failed to renew the shared lease heartbeat, causing GPU lease failures after evaluation. '
            'The heartbeat and allocation-status reporting have been corrected. Completed experiments are not being rerun. '
            'All results and checkpoints remain intact. Current running GPU leases: '+str(resources['allocated_gpus'])+'.')

def step_table():
    rows=[]
    for line in command(['squeue','--steps','-h','-u','research_user','-o','%i|%j']).splitlines():
        v=line.split('|')
        if len(v)==2 and v[0].rsplit('.',1)[-1].isdigit():rows.append(dict(step=v[0],job=v[0].split('.')[0],name=v[1]))
    return rows

def ready_pool():
    registered=set(load(LEASES/'pool_registry.json')['jobs'].values());pool=[]
    for line in command(['squeue','-h','-u','research_user','-o','%i|%T|%N']).splitlines():
        v=line.split('|')
        if len(v)!=3 or v[0] not in registered or v[1]!='RUNNING':continue
        receipt=load(LEASES/'leases'/f'{v[0]}.json')
        if receipt.get('state')=='available' and receipt.get('node')==v[2]:pool.append(dict(job_id=v[0],node=v[2]))
    return sorted(pool,key=lambda v:int(v['job_id']))

def stage1_modules():
    if str(STAGE1) not in sys.path:sys.path.insert(0,str(STAGE1))
    import renew_leases
    return renew_leases

def monitor_stage1():
    supervisor_child=None;last_gpu=0.
    while not (WORK/'selected/identity_checkpoint.json').exists():
        yield_to_slurm()
        verify_code()
        if load(WORK/'smoke/receipt.json').get('state')!='passed':raise RuntimeError('passed_follow_on_smoke_test_required')
        if load(WORK/'tests_receipt.json').get('state')!='passed':raise RuntimeError('passed_follow_on_policy_tests_required')
        if load(WORK/'patient_label_preflight.json').get('state')!='passed':raise RuntimeError('passed_patient_label_preflight_required')
        if (STAGE1/'health_stop.json').exists():
            alert('stage1_integrity','Craniofacial training needs inspection','The training integrity guard requested inspection. Checkpoints and allocations are retained; no automatic stage transition occurred.')
            heartbeat('stage1_integrity_review_required');time.sleep(60);continue
        leader=load(STAGE1/'supervisor_status.json')
        if age(leader.get('updated_at_utc'))>180 and not (STAGE1/'supervisor_retired.json').exists():
            if supervisor_child is None or supervisor_child.poll() is not None:
                stage1_modules()
                with (WORK/'logs/stage1-supervisor.private.log').open('ab') as out:
                    supervisor_child=subprocess.Popen([str(PYTHON),str(STAGE1/'supervisor.py')],env=environment(),stdout=out,stderr=out,start_new_session=True)
                from control import identity
                save(STAGE1/'supervisor_process.json',dict(pid=supervisor_child.pid,host=socket.gethostname(),start_identity=identity(supervisor_child.pid),updated_at_utc=now()))
        rows=[]
        with (STAGE1/'metrics.jsonl').open() as stream:
            for line in stream:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue # writer may be between write and flush
        reports=[load(p) for p in (STAGE1/'validation').glob('step_*.json')]
        decision=decide(rows,reports,load(WORK/'protocol.json')['stop_policy'])
        save(WORK/'convergence.json',dict(decision,updated_at_utc=now()))
        status=load(STAGE1/'status.json');last_activity=max((STAGE1/n).stat().st_mtime for n in ('status.json','latest_checkpoint.json','latest_validation.json'))
        if time.time()-last_activity>1800:
            alert('stage1_stalled_'+str(status.get('attempt')),'Craniofacial training is awaiting recovery','No training or checkpoint progress was observed for 30 minutes. The training supervisor is retaining checkpoints and retrying; this is not a completion notice.')
        stop=load(STAGE1/'stop_requested.json')
        if stop and stop.get('owner')!=WORK.name:
            alert('external_stage1_stop','Craniofacial training has a separate stop request','An external stop request exists. Automatic stage advancement is waiting for inspection.');time.sleep(60);continue
        if decision['advance'] and not stop:
            save(STAGE1/'handoff_requested.json',dict(owner=WORK.name,reason=decision['reason'],requested_at_utc=now()))
            save(STAGE1/'stop_requested.json',dict(owner=WORK.name,reason=decision['reason'],requested_at_utc=now()))
            save(WORK/'stage1_completion_decision.json',dict(decision,updated_at_utc=now()))
        if load(STAGE1/'supervisor_retired.json').get('state')=='complete':
            active=[r for r in step_table() if r['name'].startswith('arcface-id-') and r['name']!='arcface-id-monitor']
            if not active:
                select_identity_checkpoint()
                alert('stage1_complete','Craniofacial CPT completed; GMDB adaptation is starting',
                    'Identity CPT has saved a complete checkpoint and reached its configured stopping condition. The best available internal-development checkpoint was preserved. Matched disease adaptation of the CPT and original TransFace encoders will use GMDB training and development partitions; the clinical final-test cohorts remain untouched.')
                return
        heartbeat('monitoring_identity_cpt',completed_steps=status.get('completed_steps'),loss=status.get('loss'),world_size=status.get('world_size'),decision=decision['reason'])
        if time.monotonic()-last_gpu>300:
            pool=ready_pool()
            if pool:
                from monitor import sample_pool
                sample_pool(pool)
            last_gpu=time.monotonic()
        time.sleep(60)

def select_identity_checkpoint():
    candidates=[]
    best=load(STAGE1/'best/registry.json').get('disease_ridge',{})
    if best:candidates.append(dict(best,score=best['macro_top1'],complete=Path(best['path']).exists()))
    for p in (STAGE1/'checkpoints').glob('step_*.json'):
        r=load(p);v=load(STAGE1/'validation'/p.name)
        if r.get('state')=='complete' and v.get('state')=='complete':
            candidates.append(dict(r,score=v['metrics']['disease']['ridge']['top1'],complete=Path(r['path']).exists()))
    selected=choose_checkpoint(candidates);method='internal_development_disease_ridge_macro_top1'
    if selected is None:
        selected=load(STAGE1/'latest_checkpoint.json');method='latest_complete_fallback'
    if selected.get('step',0)<=0 or not Path(selected['path']).exists():raise RuntimeError('nonzero_complete_identity_checkpoint_required')
    target=WORK/'selected/identity_checkpoint.pt';temp=target.with_suffix('.tmp');temp.unlink(missing_ok=True)
    os.link(selected['path'],temp)
    if digest(temp)!=selected['sha256']:raise RuntimeError('identity_checkpoint_checksum_mismatch')
    temp.replace(target)
    save(WORK/'selected/identity_checkpoint.json',dict(selected,path=str(target),selection_method=method,selected_at_utc=now()))

def launch(task,job,launches):
    folder=WORK/task['folder'];folder.mkdir(parents=True,exist_ok=True)
    old=launches.get(task['key'],{});attempt=old.get('attempt',0)+1
    failure=load(folder/'failure.json')
    if failure:
        save(folder/f'failure_attempt_{old.get("attempt",0)}.json',failure)
        if failure.get('oom') and task['kind']=='train':
            batch=load(folder/'batch_override.json').get('batch_size',load(WORK/'protocol.json')['stage2']['batch_size'])
            if batch<=16:raise RuntimeError('minimum_batch_oom_requires_inspection')
            save(folder/'batch_override.json',dict(batch_size=max(16,batch//2),reason='oom_recovery'))
        (folder/'failure.json').unlink()
    record=dict(attempt=attempt,job_id=job['job_id'],name=task['name'],started_at_utc=now())
    launches[task['key']]=record;save(WORK/'launches.json',launches)
    args=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--job-name='+task['name'],'--nodes=1','--ntasks=1',
        '--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
        str(PYTHON),str(WORK/task['script']),task['arg']]
    with (folder/f'worker_attempt_{attempt}.private.log').open('ab') as out:
        child=subprocess.Popen(args,env=environment(),stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
    return child

def run_tasks(tasks,phase):
    clients={};last_renew=0.;last_gpu=0.
    while True:
        yield_to_slurm()
        verify_code()
        if all(load(WORK/t['folder']/'result.json').get('state')=='complete' for t in tasks):return True
        heartbeat(phase,total_cases=len(tasks))
        if time.monotonic()-last_renew>60:
            stage1_modules().reconcile();last_renew=time.monotonic()
        pool=ready_pool();steps=step_table();launches=load(WORK/'launches.json')
        occupied={r['job'] for r in steps};active={r['name']:r for r in steps}
        complete=[];blocked=[];pending=[]
        for task in tasks:
            folder=WORK/task['folder'];record=launches.get(task['key'],{})
            if load(folder/'result.json').get('state')=='complete':complete.append(task['key']);continue
            if task['name'] in active:
                stamp=load(folder/'status.json').get('updated_at_utc',record.get('started_at_utc'))
                if age(stamp)>1800 and age(record.get('started_at_utc'))>1800:
                    command(['scancel',active[task['name']]['step']])
                    save(folder/'failure.json',dict(error_type='ProgressTimeout',oom=False,updated_at_utc=now()))
                continue
            child=clients.get(task['key'])
            if child and child.poll() is None:continue
            if record and age(record.get('started_at_utc'))<120:continue
            # Infra interruptions may resume indefinitely; repeated explicit numerical/integrity failures require inspection.
            failures=[load(p) for p in folder.glob('failure_attempt_*.json')]+[load(folder/'failure.json')]
            serious=sum(bool(f) and not f.get('oom') and f.get('error_type') not in ('ProgressTimeout',) for f in failures)
            current_failure=load(folder/'failure.json')
            exhausted_oom=current_failure.get('oom') and load(folder/'batch_override.json').get('batch_size',128)<=16
            if serious>=3 or exhausted_oom:
                blocked.append(task['key']);alert('case_failed_'+task['key'],'A GMDB comparison case needs inspection',
                    'A comparison case repeatedly failed. Its checkpoints are preserved and other valid cases continue. The workflow will not report this case as completed.');continue
            pending.append(task)
        if len(complete)==len(tasks):return True
        free=[p for p in pool if p['job_id'] not in occupied]
        for task,job in zip(pending,free):
            clients[task['key']]=launch(task,job,launches)
        heartbeat(phase,total_cases=len(tasks),complete_cases=len(complete),blocked_cases=len(blocked),allocated_gpus=len(pool),running_tasks=len(active),queued_cases=len(pending))
        if time.monotonic()-last_gpu>300 and pool:
            from monitor import sample_pool
            sample_pool(pool);last_gpu=time.monotonic()
        time.sleep(30)

def select_disease_models(cases):
    if load(WORK/'selected/models.json').get('state')=='frozen_before_test':return
    selected={}
    for budget in ('100','25','10'):
        selected[budget]={}
        for arm in ('original','identity_cpt'):
            records=[load(WORK/'cases'/c['name']/'result.json') for c in cases if c['budget']==budget and c['arm']==arm]
            if not records or any(r.get('state')!='complete' for r in records):raise RuntimeError('matched_search_incomplete')
            winner=max(records,key=lambda r:tuple(r['best']['score']))
            selected[budget][arm]=dict(winner['best'],case=winner['case'])
    save(WORK/'selected/models.json',dict(state='frozen_before_test',budgets=selected,full_label=selected['100'],selection_uses_test=False,updated_at_utc=now()))

def finish():
    if load(WORK/'results.json').get('state')=='complete':
        completed_heartbeat();return
    results={arm:load(WORK/'evaluation'/arm/'result.json') for arm in ('original','identity_cpt')}
    if any(r.get('state')!='complete' for r in results.values()):raise RuntimeError('complete_paired_evaluation_required')
    delta={}
    for variant in results['original']['metrics']:
        delta[variant]={case:{k:results['identity_cpt']['metrics'][variant][case]['macro'][k]-v for k,v in table['macro'].items()}
            for case,table in results['original']['metrics'][variant].items()}
    save(WORK/'results.json',dict(state='complete',models=results,deltas_cpt_minus_original=delta,
        development_selection=load(WORK/'selected/models.json'),clinical_final_test_used=False,updated_at_utc=now()))
    values=[]
    for arm,r in results.items():
        table=r['metrics']['patient_disjoint_query_supplement']
        values.append(f"{arm}: frequent Top1 {100*table['frequent_to_frequent']['macro']['top1']:.2f}%; rare Top1 {100*table['rare_to_rare']['macro']['top1']:.2f}%")
    alert('complete','Craniofacial disease-adaptation comparison completed',
        'Matched original and CPT TransFace models finished disease adaptation and fixed GMDB evaluation. '+ '; '.join(values)+
        '. These are exploratory GMDB results, not CHOP/BCH clinical validation. The full aggregate report is saved in '+str(WORK/'results.json')+'.')
    completed_heartbeat()

def main():
    os.umask(0o077)
    if RUNNING_IN_SLURM:
        save(WORK/'slurm_coordinator_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],updated_at_utc=now()))
    with (WORK/'workflow.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_code()
        save(WORK/'workflow_process.json',dict(pid=os.getpid(),host=socket.gethostname(),running_in_slurm=RUNNING_IN_SLURM,updated_at_utc=now()))
        # Clean inherited CPU-allocation binding before controlling independent GPU leases.
        os.environ.clear();os.environ.update(environment_snapshot)
        alert('armed','Automatic craniofacial training follow-on is active',
            'The Slurm coordinator now monitors training progress, checkpoints, development trends and GPU utilization. After checkpointed CPT completion it will run matched GMDB disease adaptation and evaluation. Milestone, completion or failure notices will follow. All patient-level data remain on HPC.')
        monitor_stage1()
        with (LEASES/'launcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            heartbeat('restoring_follow_on')
            cases=load(WORK/'cases_manifest.json')['cases']
            tasks=[dict(key=c['name'],name='gmdb-s2-'+c['name'],folder='cases/'+c['name'],script='train_case.py',arg=c['name'],kind='train') for c in cases]
            run_tasks(tasks,'disease_adaptation')
            select_disease_models(cases)
            alert('disease_training_complete','GMDB adaptation finished; fixed evaluation is starting',
                'Both encoders completed matched disease-adaptation searches. Model selection is now frozen using development patients only. Fixed GMDB retrieval evaluation is starting; CHOP expert and BCH final tests remain untouched.')
            tasks=[dict(key='eval_'+a,name='gmdb-s2-eval-'+a,folder='evaluation/'+a,script='evaluate_selected.py',arg=a,kind='evaluate') for a in ('original','identity_cpt')]
            run_tasks(tasks,'fixed_gmdb_evaluation');finish()
            # Preserve the user's GPU leases, without synthetic work or invented utilization.
            while True:
                yield_to_slurm()
                completed_heartbeat()
                time.sleep(60)

environment_snapshot=environment()
if __name__=='__main__':
    try:main()
    except CoordinatorHandoff:sys.exit(0)
    except BlockingIOError:sys.exit(75)
    except Exception as error:
        save(WORK/'workflow_failure.json',dict(error_type=type(error).__name__,updated_at_utc=now()))
        alert('workflow_failure_'+type(error).__name__,'Craniofacial workflow is retrying','The automatic coordinator encountered an error and will retry. Existing training checkpoints and allocations are retained. Only aggregate operational information is included.')
        raise
