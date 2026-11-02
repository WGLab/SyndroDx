"""Durable development gate followed by matched multimodal diagnosis experiments."""
import fcntl
import os
import subprocess
import time
from common import *
from supervisor import query,launch,stop_clients

def wait_all(clients):
    while any(c.poll() is None for c in clients):
        if any(c.poll() not in (None,0) for c in clients):
            stop_clients(clients);raise RuntimeError('followon_worker_failed')
        time.sleep(5)
    if any(c.returncode!=0 for c in clients):raise RuntimeError('followon_worker_failed')

def main():
    lock=(WORK/'followon.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(WORK/'training_complete.json').get('state')!='complete':raise RuntimeError('alignment_completion_required')
    while not query():time.sleep(15)
    if not load(WORK/'alignment_gate.json'):
        wait_all([launch(query()[0],'check_alignment.py',[],'align-validation-gate')])
    gate={}
    for _ in range(24):
        gate=load(WORK/'alignment_gate.json')
        if gate.get('state'):break
        time.sleep(5)
    if not gate.get('state'):raise RuntimeError('validation_receipt_visibility_timeout')
    if gate.get('state')!='passed':
        save(WORK/'followon_status.json',dict(state='alignment_needs_inspection',gate=gate,updated_at_utc=now()))
        notify('alignment_gate_failed','Alignment did not pass the shuffled-pair development check; fusion has not started. Review the aggregate gate receipt on HPC. Allocations preserved.')
        return
    with (WORK/'logs/prepare_fusion.private.log').open('ab') as log:
        subprocess.run([str(PYTHON),str(WORK/'prepare_fusion.py')],env=environment(),stdout=log,stderr=log,check=True)
    for mode in ['fusion','hpo','image']:
        failures=0;attempt=0
        while load(WORK/'fusion'/mode/'result.json').get('state')!='complete':
            jobs=query()
            if not jobs:time.sleep(15);continue
            attempt+=1;save(WORK/'fusion_resize.json',dict(world_size=len(jobs)))
            env=dict(WORLD_SIZE=str(len(jobs)),MASTER_ADDR=jobs[0]['node'],MASTER_PORT=str(34000+attempt),LOCAL_RANK='0')
            clients=[launch(j,'train_fusion.py',[mode],f'align-diagnosis-{mode}-a{attempt}-r{rank}',dict(env,RANK=str(rank))) for rank,j in enumerate(jobs)]
            save(WORK/'followon_status.json',dict(state='training',mode=mode,world_size=len(jobs),jobs=jobs,updated_at_utc=now()))
            while any(c.poll() is None for c in clients):
                if any(c.poll() not in (None,0) for c in clients):stop_clients(clients);break
                current=query()
                save(WORK/'fusion_resize.json',dict(world_size=len(current)))
                time.sleep(10)
            if any(c.poll()!=0 for c in clients):
                failures+=1
                if failures>=2:raise RuntimeError('repeated_diagnosis_training_failure')
            time.sleep(3)
    results={m:load(WORK/'fusion'/m/'result.json') for m in ['fusion','hpo','image']}
    fusion=results['fusion']['best'];baselines=[results[m]['best']['macro_top5'] for m in ['hpo','image']]
    save(WORK/'fusion_results.json',dict(state='complete',alignment_gate=gate,models=results,
        fusion_beats_both_unimodal_development_top5=fusion['macro_top5']>max(baselines),
        clinical_test_used=False,next_action='independent replication and clinical holdout evaluation design' if fusion['macro_top5']>max(baselines) else 'inspect modality contributions before scaling',updated_at_utc=now()))
    save(WORK/'followon_status.json',dict(state='complete',results_file=str(WORK/'fusion_results.json'),updated_at_utc=now()))
    notify('fusion_complete','Alignment validation and matched fusion/HPO-only/image-only diagnosis probes completed. Aggregate results: '+str(WORK/'fusion_results.json')+'. CHOP expert/BCH tests were not evaluated. GPU allocations preserved.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'followon_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('fusion_failure','Follow-on controller requires inspection. Private logs remain on HPC. Allocations preserved.')
        except Exception:pass
        raise
