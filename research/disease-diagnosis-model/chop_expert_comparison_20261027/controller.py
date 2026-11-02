"""Complete the exposure audit, freeze four checkpoints and score one expert cohort."""
import os,sys,json,subprocess,time,fcntl,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BENCH=PROJECT/'gestaltmatcher_arc_comparison_20261027'
sys.path.insert(0,str(BENCH))
from controller import jobs,launch,runtime
from common import PYTHON,load,save,now

def free_jobs():
    output=subprocess.run([str(runtime.SLURM/'squeue'),'--steps','-h','-u','research_user','-o','%i'],env=runtime.environment(),capture_output=True,text=True,check=True).stdout
    busy={s.split('.')[0] for s in output.splitlines() if s.rsplit('.',1)[-1].isdigit()}
    return [j for j in jobs() if j['id'] not in busy]

def run_cpu(script):
    with (ROOT/'logs'/(script+'.private.log')).open('ab') as log:
        subprocess.run([str(PYTHON),str(ROOT/script)],env=runtime.environment(),stdout=log,stderr=log,check=True)

def main():
    os.umask(0o077);lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not free_jobs():time.sleep(20)
    clients={}
    original=BENCH/'cases/transface_original/result.json'
    if load(original).get('state')!='complete':
        clients['original_gmdb']=launch(free_jobs()[0],'train_case.py','transface_original','chop-expert-original-gmdb-control')
    if not (ROOT/'audit_summary.json').is_file():run_cpu('audit.py')
    save(ROOT/'status.json',dict(state='near_duplicate_audit',initial_audit=load(ROOT/'audit_summary.json'),updated_at_utc=now()))
    if not (ROOT/'release.json').is_file():run_cpu('near_audit.py')
    release=load(ROOT/'release.json')
    if release.get('retrieval_evaluable_patients',0)<10:raise RuntimeError('insufficient_audit_released_expert_patients')
    while clients:
        if any(c.poll() not in (None,0) for c in clients.values()):raise RuntimeError('GMDB_control_training_failed')
        if all(c.poll()==0 for c in clients.values()):break
        save(ROOT/'status.json',dict(state='waiting_for_matched_original_GMDB_control',updated_at_utc=now()));time.sleep(15)
    # Allow compute-node receipt writes to become visible on this shared filesystem.
    for _ in range(24):
        if load(original).get('state')=='complete':break
        time.sleep(5)
    if load(original).get('state')!='complete':raise RuntimeError('matched_original_control_incomplete')
    arms=['original','cpt','original_gmdb','cpt_gmdb'];clients={};attempts={}
    for case in ['transface_original','transface']:
        chosen=load(BENCH/'cases'/case/'best.json')
        if hashlib.sha256(Path(chosen['path']).read_bytes()).hexdigest()!=chosen['sha256']:raise RuntimeError('selected_GMDB_checkpoint_changed')
    save(ROOT/'frozen_protocol.json',dict(state='frozen_before_expert_scoring',arms=arms,
        original_GMDB_checkpoint=load(BENCH/'cases/transface_original/best.json'),CPT_GMDB_checkpoint=load(BENCH/'cases/transface/best.json'),
        CPT_checkpoint=load(PROJECT/'gmdb_supervised_transface_20261026/selected/identity_checkpoint.json'),
        shared_expert_release=release,training_images_equal_for_GMDB_controls=True,expert_labels_used_to_select_models=False,updated_at_utc=now()))
    while True:
        done=[a for a in arms if load(ROOT/(a+'_receipt.json')).get('state')=='complete']
        if len(done)==4:break
        for arm,c in list(clients.items()):
            if c.poll() is not None:
                if c.returncode and attempts[arm]>=2:raise RuntimeError('repeated_expert_feature_failure')
                del clients[arm]
        pool=free_jobs()
        for arm,job in zip([a for a in arms if a not in done and a not in clients],pool):
            attempts[arm]=attempts.get(arm,0)+1
            clients[arm]=launch(job,str(ROOT/'infer.py'),arm,'chop-expert-'+arm+'-'+str(attempts[arm]))
        save(ROOT/'status.json',dict(state='extracting_four_models',completed=done,active=list(clients),updated_at_utc=now()));time.sleep(15)
    run_cpu('score.py')
    save(ROOT/'status.json',dict(state='complete',results_file=str(ROOT/'results.json'),updated_at_utc=now()))
    runtime.notify('CHOP_expert_four_model_comparison_complete','The audit and fixed four-model CHOP expert-source comparison completed. Only aggregate results are stored at '+str(ROOT/'results.json')+'. See the release limitations before clinical interpretation. GPU allocations preserved.')

if __name__=='__main__':
    try:main()
    except Exception as e:save(ROOT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
