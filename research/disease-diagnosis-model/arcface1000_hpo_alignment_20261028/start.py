"""Give alignment two GPUs at a seed checkpoint, then resume the seed arm."""
import os,sys,time,subprocess
from pathlib import Path
from common import *
def main():
    os.umask(0o077)
    with (WORK/'logs/preparation.private.log').open('ab') as log:
        subprocess.run([str(PYTHON),str(WORK/'prepare.py')],env=environment(),stdout=log,stderr=log,check=True)
    selection=load(PROJECT/'arcface_transfer_evaluation_20261028/selection.json')
    assert selection['step']==1000 and digest(selection['path'])==selection['sha256']
    save(WORK/'tests_receipt.json',dict(state='passed',scope='Reuse previously synthetic-tested semantic loss and DDP training; new frozen visual checkpoint integrity verified',updated_at_utc=now()))
    queued=load(PROJECT/'arcface_transfer_evaluation_20261028/seed_queue_status.json')
    paused=None
    if queued.get('path'):
        roundpath=Path(queued['path']);arms=load(roundpath/'protocol.json').get('arms',{})
        for arm in arms:
            path=roundpath/arm/'pause_requested.json'
            if not path.exists():
                save(path,dict(reason='checkpointed_temporary_yield_for_two_gpu_alignment',updated_at_utc=now()));paused=path;break
    with (WORK/'logs/supervisor.private.log').open('ab') as log:
        child=subprocess.Popen([str(PYTHON),str(WORK/'supervisor.py')],env=environment(),stdout=log,stderr=log)
        while child.poll() is None:
            text=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%j'],env=environment(),capture_output=True,text=True,check=True).stdout
            if paused and any(n.startswith('align-') for n in text.splitlines()):
                paused.unlink(missing_ok=True);paused=None
            time.sleep(10)
        if paused:paused.unlink(missing_ok=True)
        if child.returncode:raise RuntimeError('alignment_supervisor_failed')
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'startup_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
