"""Persist the one-time checkpoint handoff using the existing cron watchdog."""
import fcntl
import subprocess
from runtime import ROOT, SLURM, load, save, now, environment, notify


def ensure():
    request = load(ROOT / 'training/checkpoint_batch_handoff.json')
    if request.get('state') != 'waiting_for_checkpoint':
        return
    with (ROOT / 'training/checkpoint_batch_submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = ROOT / 'training/checkpoint_batch_submission.json'
        status = load(status_path)
        if status.get('job_id'):
            query = subprocess.run([str(SLURM / 'squeue'), '-h', '-j', status['job_id'], '-o', '%T'],
                                   capture_output=True, text=True, timeout=25)
            if query.returncode == 0 and query.stdout.strip():
                return
        if status.get('attempts', 0) >= 3:
            notify('Checkpoint batch watcher requires attention', 'SSL batch watcher requires attention',
                   'The local checkpoint watcher could not remain active after retries. Existing training and GPU reservations are preserved; private diagnostics remain on HPC.')
            return
        body = ROOT / 'training/checkpoint_batch_watch.sbatch'
        body.write_text('#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true DO_NOT_TRACK=1\nexec /usr/bin/python3 ' + str(ROOT / 'checkpoint_batch_handoff.py') + '\n')
        command = [str(SLURM / 'sbatch'), '--parsable', '--partition=defq', '--job-name=ssl-next-checkpoint-batch',
                   '--cpus-per-task=1', '--mem=1G', '--time=04:00:00', '--chdir=' + str(ROOT),
                   '--output=' + str(ROOT / 'training/checkpoint_batch_watch.private.log'), str(body)]
        result = subprocess.run(command, env=environment(), capture_output=True, text=True, check=True, timeout=30)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('checkpoint_batch_watch_submission_failed')
        save(status_path, dict(state='submitted', job_id=job, attempts=status.get('attempts', 0) + 1,
                               target_checkpoint_iteration=request['target_checkpoint_iteration'], submitted_at_utc=now()))


if __name__ == '__main__':
    ensure()
