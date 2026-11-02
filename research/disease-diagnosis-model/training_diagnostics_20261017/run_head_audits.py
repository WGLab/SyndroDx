"""Use two idle existing allocations for bounded offline diagnostics."""
import contextlib
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
WORK = Path('/opt/craniofacial/disease-diagnosis-model/training_diagnostics_20261017')
sys.path.insert(0, str(BASE))
from common import V1, SLURM, PYTHON, environment, load, save, now, notify
from allocations import active_allocations
sys.path.insert(0, str(V1))
from runtime import claim_jobs


def main(kinds=('initial', 'final')):
    os.umask(0o077)
    pool, _ = active_allocations()
    result = subprocess.run([str(SLURM/'squeue'), '-h', '--steps', '-u', 'research_user', '-o', '%i'],
                            capture_output=True, text=True, check=True, env=environment(), timeout=30)
    busy = {line.split('.')[0] for line in result.stdout.splitlines()
            if '.' in line and not line.endswith(('.batch', '.extern'))}
    pool = [allocation for allocation in pool if allocation['job_id'] not in busy][:len(kinds)]
    if len(pool) < len(kinds):
        raise RuntimeError('enough_idle_existing_allocations_required')
    locks = claim_jobs([item['job_id'] for item in pool])
    clients = []
    try:
        for kind, allocation in zip(kinds, pool):
            command = [str(SLURM/'srun'), '--jobid='+allocation['job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=4', '--mem=40G', '--gpus-per-task=1', '--gpu-bind=single:1',
                       '--cpu-bind=none', '--exact', '--exclusive', '--mpi=none', '--time=00:25:00',
                       str(PYTHON), str(WORK/'head_audit.py'), '--kind', kind]
            with (WORK/f'head_audit_{kind}.private.log').open('ab') as output:
                clients.append(subprocess.Popen(command, env=environment(), stdin=subprocess.DEVNULL,
                                                stdout=output, stderr=output))
        save(WORK/controller_name(kinds), {'state':'running', 'GPU_workers':len(kinds),
              'existing_parent_allocations_retained':True, 'started_at_utc':now()})
        deadline = time.monotonic()+28*60
        while any(client.poll() is None for client in clients):
            if time.monotonic() > deadline:
                raise RuntimeError('head_diagnostic_time_limit')
            time.sleep(10)
        codes = [client.returncode for client in clients]
        state = 'complete' if all(code == 0 for code in codes) else 'failed'
        save(WORK/controller_name(kinds), {'state':state, 'worker_exit_codes':codes,
              'existing_parent_allocations_retained':True, 'completed_at_utc':now()})
        notify('projection-head audit '+','.join(kinds)+' '+state, 'Projection-head diagnostic '+state,
               'The bounded local projection-head diagnostic has finished. Private reports remain on HPC. This is a diagnostic, not evidence of clinical superiority.')
        if state != 'complete':
            raise RuntimeError('head_diagnostic_failed')
    finally:
        for client in clients:
            if client.poll() is None:
                client.terminate()
        for client in clients:
            with contextlib.suppress(subprocess.TimeoutExpired):
                client.wait(timeout=15)
        for lock in locks:
            lock.close()


def controller_name(kinds):
    return 'head_audit_controller.json' if tuple(kinds) == ('initial', 'final') else 'head_audit_'+'_'.join(kinds)+'_controller.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinds', choices=['initial', 'final', 'warmup200', 'warmup1000'], nargs='+', default=['initial', 'final'])
    args = parser.parse_args()
    try:
        main(args.kinds)
    except Exception as error:
        save(WORK/controller_name(args.kinds), {'state':'failed', 'error_type':type(error).__name__,
              'updated_at_utc':now(), 'existing_parent_allocations_retained':True})
        raise
