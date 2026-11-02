"""Submit eight independent H100 leases; never accesses patient data."""
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from common import SLURM, environment, load, now, save

NAME = 'cranio-unfreeze-pool-20261019'


def main():
    WORK.mkdir(exist_ok=True)
    with (WORK/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queued = subprocess.run([str(SLURM/'squeue'), '-h', '-r', '-u', 'research_user', '-n', NAME,
                                 '-o', '%i|%T'], env=environment(), capture_output=True,
                                text=True, check=True, timeout=30)
        if queued.stdout.strip():
            print(json.dumps({'state':'existing_request_retained', 'jobs':queued.stdout.splitlines()}))
            return
        previous = load(WORK/'submission.json')
        if previous.get('array_job_id'):
            raise RuntimeError('previous_request_ended_review_before_resubmitting')
        # Keep the historical receipts before the unchanged allocation script
        # updates the active slots consumed by the existing training helpers.
        backup = WORK/'previous_allocation_receipts'
        backup.mkdir(exist_ok=True)
        for slot in range(8):
            source = BASE/'allocations'/f'slot_{slot:02d}.json'
            if source.exists() and not (backup/source.name).exists():
                shutil.copy2(source, backup/source.name)
        command = [str(SLURM/'sbatch'), '--parsable', '--partition=gpu-xe9680q',
                   '--job-name='+NAME, '--array=0-7', '--gres=gpu:h100:1',
                   '--cpus-per-task=4', '--mem=72G', '--time=10-00:00:00',
                   '--chdir='+str(BASE), '--output='+str(WORK/'allocation_%A_%a.operational.log'),
                   str(BASE/'allocate_single.sbatch')]
        result = subprocess.run(command, env=environment(), capture_output=True,
                                text=True, timeout=45)
        if result.returncode:
            save(WORK/'submission_failure.json', {'state':'submission_failed',
                 'scheduler_error':result.stderr.strip(), 'checked_at_utc':now()})
            print(json.dumps({'state':'submission_failed', 'scheduler_error':result.stderr.strip()}))
            raise SystemExit(1)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('unrecognized_scheduler_receipt')
        receipt = {'state':'submitted', 'array_job_id':job, 'job_name':NAME,
                   'requested_GPUs':8, 'GPUs_per_independent_allocation':1,
                   'requested_duration_days':10, 'partition':'gpu-xe9680q',
                   'GPU_type':'h100', 'CPU_cores_per_allocation':4,
                   'host_memory_per_allocation_GiB':72,
                   'old_receipts_preserved':True, 'patient_data_accessed':False,
                   'submitted_at_utc':now()}
        save(WORK/'submission.json', receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
