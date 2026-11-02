"""Continue useful, fixed-budget ArcFace seed rounds on available allocations."""
import os,sys,time,subprocess,shutil,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'transface_objective_comparison_20261028'))
from matched_runtime import *
HOME_RUN=PROJECT/'arcface_extra_seeds_20261028'

def main():
    os.umask(0o077)
    import fcntl
    lock=(HOME_RUN/'continuous_seeds.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(HOME_RUN/'continuous_policy.json',dict(enabled=True,stop_only_on_user_instruction=True,
        activity='independent low-LR ArcFace CPT seed rounds, fixed development evaluation; no repeated test selection',
        allocations_preserved=True,checkpoint_recovery=True,updated_at_utc=now()))
    round_number=0
    while not (HOME_RUN/'stop_continuous_seeds.json').exists():
        target=PROJECT/f'arcface_extra_seed_round_{round_number:04d}_20261028'
        if load(target/'status.json').get('state') in ['complete','inspection_required']:
            round_number+=1;continue
        if len(free_jobs())<1:
            save(HOME_RUN/'seed_queue_status.json',dict(state='waiting_for_available_allocations',round=round_number,updated_at_utc=now()));time.sleep(30);continue
        if not (target/'protocol.json').exists():
            target.mkdir(exist_ok=True)
            for p in ROOT.glob('*.py'):shutil.copy2(p,target/p.name)
            # Separate rendezvous ports from the original and fill seed queues.
            controller = target/'controller.py'
            controller.write_text(controller.read_text().replace('port=42000+', 'port=57000+'))
            runtime=(target/'matched_runtime.py').read_text()
            arms=[f'arcface_seed_{20262100+round_number*4+i}' for i in range(1)]
            runtime=re.sub(r'^ARMS = .*$', 'ARMS = '+repr(arms),runtime,flags=re.M)
            (target/'matched_runtime.py').write_text(runtime)
            train=(target/'train.py').read_text().replace("seed=p['seed'];", "seed=p['seed']+ARMS.index(arm);")
            (target/'train.py').write_text(train)
            # Freeze the independently chosen checkpoint-selection rule; test results never drive these runs.
            p=load(ROOT/'protocol.json');p['seed']=20262100+round_number*4
            low=p['arms']['arcface_lr5e7'];p['arms']={a:dict(low) for a in arms}
            p['experiment']='independent seed replication of low-LR patient identity ArcFace CPT'
            p['test_feedback_used_for_selection']=False;p['created_at_utc']=now()
            (target/'private').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'private/gmdb_patients.json',target/'private/gmdb_patients.json')
            save(target/'audit.json',load(ROOT/'audit.json'));save(target/'protocol.json',p)
            for a in arms:
                for n in ['logs','evaluations','checkpoints']:(target/a/n).mkdir(parents=True,exist_ok=True)
            (target/'logs').mkdir(exist_ok=True)
        save(HOME_RUN/'seed_queue_status.json',dict(state='running_seed_round',round=round_number,path=str(target),updated_at_utc=now()))
        with (target/'logs/controller.private.log').open('ab') as log:
            code=subprocess.run([str(PYTHON),str(target/'controller.py')],env=environment(),stdout=log,stderr=log).returncode
        if code:
            save(HOME_RUN/'seed_queue_failure.json',dict(round=round_number,returncode=code,updated_at_utc=now()))
            notify('seed queue requires inspection','Independent ArcFace seed controller failed; checkpoints and allocations retained.');return
        round_number+=1

if __name__=='__main__':main()
