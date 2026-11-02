"""Run matched diagnosis adapter arms after the visual-conditioning gate."""
import fcntl,os,subprocess,time
from runtime import *
from controller import jobs,launch,stop,wait_receipt
def main():
    os.umask(0o077);lock=(WORK/'followon.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not load(WORK/'training_complete.json'):time.sleep(15)
    if not load(WORK/'training_complete.json').get('visual_conditioning_gate'):raise RuntimeError('visual_conditioning_gate_failed')
    if load(WORK/'sft_tests_receipt.json').get('state')!='passed':raise RuntimeError('SFT_gradient_tests_required')
    while any(line.strip().split('|')[-1].startswith('cranio-qwen-train-') for line in subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%i|%j'],env=env(),capture_output=True,text=True,check=True).stdout.splitlines()):time.sleep(10)
    with (WORK/'logs/prepare-sft.private.log').open('ab') as f:subprocess.run([str(PYTHON),str(WORK/'prepare_sft.py')],env=env(),stdout=f,stderr=f,check=True)
    for arm in load(WORK/'sft_protocol.json')['arms']:
        out=WORK/'sft'/arm;attempt=load(out/'attempt.json').get('attempt',0);failures=0
        while not load(out/'training_complete.json'):
            pool=jobs()
            if not pool:time.sleep(20);continue
            attempt+=1;save(out/'attempt.json',dict(attempt=attempt));save(out/'resize_requested.json',dict(world_size=len(pool)))
            batch=load(out/'batch_override.json').get('batch',2)
            clients=[launch(j,'sft.py',[arm],f'cranio-qwen-sft-{arm}-a{attempt}-r{r}',dict(RANK=str(r),WORLD_SIZE=str(len(pool)),LOCAL_RANK='0',MASTER_ADDR=pool[0]['node'],MASTER_PORT=str(33500+attempt+(100 if arm=='hpo' else 0)),CRANIO_ATTEMPT=str(attempt),CRANIO_BATCH=str(batch))) for r,j in enumerate(pool)]
            save(WORK/'sft_active_training.json',dict(arm=arm,attempt=attempt,jobs=pool,world_size=len(pool),updated_at_utc=now()))
            while any(c.poll() is None for c in clients):
                if any(c.poll() not in (None,0) for c in clients):stop(clients);break
                available=jobs();save(out/'resize_requested.json',dict(world_size=len(available)))
                save(WORK/'followon_status.json',dict(state='disease_SFT_running',arm=arm,world_size=len(pool),available_gpus=len(available),progress=load(out/'status.json'),updated_at_utc=now()))
                time.sleep(20)
            if all(c.returncode==0 for c in clients):
                resize=load(out/'resize_requested.json').get('world_size',len(pool))!=len(pool)
                wait_receipt(out/('latest_checkpoint.json' if resize else 'training_complete.json'),180)
            errors=[load(out/'workers'/f'{attempt}_{r}.failure.json') for r in range(len(pool))]
            if any(errors) or any(c.returncode for c in clients):
                failures+=1
                if any(e.get('oom') for e in errors) and batch>1:save(out/'batch_override.json',dict(batch=batch//2))
                elif failures>=2:raise RuntimeError('repeated_SFT_worker_failure')
            if failures>=3:raise RuntimeError('repeated_SFT_worker_failure')
    save(WORK/'followon_status.json',dict(state='SFT_complete_ranking_pending',allocations_preserved=True,updated_at_utc=now()))
    notify('disease_SFT_complete','Qwen32B fusion and matched HPO-only diagnosis adapter training completed. Ranking evaluation will use the same held-out development patients. Private inputs remain on HPC.')
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'followon_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('diagnosis_SFT_failure','Diagnosis adapter controller needs inspection: '+type(e).__name__+'. No allocation was cancelled.')
        except Exception:pass
        raise
