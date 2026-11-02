"""Run four complementary inference shards on retained allocations."""
import ast
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
WORK=Path(__file__).resolve().parent
TRAIN=WORK.parent/'transface_vitl_cpt_20261026';SHARED=WORK.parent/'gmdb_checkpoint_eval_20261026'
sys.path.insert(0,str(TRAIN))
from control import load,save,digest,now,environment,PYTHON,SLURM,notify,WEIGHTS

def main():
    os.umask(0o077)
    with (WORK/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if load(TRAIN/'health_stop.json').get('completed_steps')!=4000:raise RuntimeError('expected_paused_cpt_checkpoint')
        for name,sha in load(TRAIN/'code_receipt.json').items():
            if digest(TRAIN/name)!=sha:raise RuntimeError('training_code_changed')
        if digest(WEIGHTS)!=load(TRAIN/'conversion_receipt.json')['model_sha256']:
            raise RuntimeError('original_weights_changed')
        from controller import resources
        pool=resources()
        if len(pool)<4:raise RuntimeError('four_retained_gpus_required')
        checkpoint=load(TRAIN/'latest_checkpoint.json')
        if checkpoint['step']!=4000 or digest(checkpoint['path'])!=checkpoint['sha256']:raise RuntimeError('checkpoint_integrity_failure')
        pinned=WORK/'checkpoint_4000.pt'
        if not pinned.exists():os.link(checkpoint['path'],pinned)
        original=load(SHARED/'protocol.json')
        p=dict(state='prepared',checkpoint_step=4000,checkpoint_path=str(pinned),checkpoint_sha256=checkpoint['sha256'],
            inputs_sha256=digest(SHARED/'inputs.private.jsonl'),split_sha256=original['split_sha256'],
            metric_driver_sha256=digest(SHARED/'evaluate_checkpoint.py'),metrics_sha256=digest(SHARED/'metrics.py'),
            model_code_sha256=digest(TRAIN/'model.py'),encoder_frozen=True,gmdb_finetuned=False,
            exploratory_repeated_test=True,external_uploads=False,created_at_utc=now())
        # Check against the already-frozen reference inputs, not merely new hashes.
        if p['inputs_sha256']!=original['inputs_sha256']:raise RuntimeError('reference_inputs_changed')
        save(WORK/'protocol.json',p)
        clients=[]
        try:
            for i,(kind,rank) in enumerate([('original',0),('original',1),('cpt4000',0),('cpt4000',1)]):
                args=[str(SLURM/'srun'),'--jobid='+pool[i]['job_id'],f'--job-name=transface-gmdb-{kind}-{rank}',
                    '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=80G','--gpus-per-task=1',
                    '--gpu-bind=single:1','--cpu-bind=none','--overlap','--exact','--mpi=none',
                    str(PYTHON),str(WORK/'worker.py'),kind,str(rank)]
                with (WORK/'logs'/f'{kind}_{rank}.private.log').open('ab') as out:
                    clients.append(subprocess.Popen(args,env=environment(),stdin=subprocess.DEVNULL,stdout=out,stderr=out))
            save(WORK/'status.json',dict(state='extracting_features',gpus=4,allocations=pool[:4],updated_at_utc=now()))
            begun=time.monotonic()
            while any(p.poll() is None for p in clients):
                if any(p.poll() not in (None,0) for p in clients):raise RuntimeError('inference_shard_failed')
                if time.monotonic()-begun>1800:raise RuntimeError('inference_deadline_exceeded')
                time.sleep(5)
            if any(p.returncode for p in clients):raise RuntimeError('inference_shard_failed')
            with (WORK/'logs/scoring.private.log').open('ab') as out:
                subprocess.run([str(PYTHON),str(WORK/'score.py')],env=environment(),stdout=out,stderr=out,check=True,timeout=1800)
            notify('TransFace GMDB diagnostic 4000 complete','TransFace GMDB diagnostic completed',
                'The original TransFace-L and 4000-step clinical CPT teacher have completed matched exploratory GMDB retrieval evaluation. Aggregate results remain on HPC. Training remains paused by the development guard; images and patient identifiers were not included in this notification.')
        finally:
            for p in clients:
                if p.poll() is None:p.terminate()

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
