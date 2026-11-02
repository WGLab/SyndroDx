"""Use the existing local cron watchdog to keep the full pipeline monitor alive."""
import fcntl
import subprocess
from runtime import ROOT, RUN, SLURM, load, save, environment, now


def ensure():
    if load(RUN / 'pipeline_evaluation_completion.json').get('state') == 'complete':
        return
    folder = ROOT / 'pipeline'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'submission.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    name = 'craniofacial-pipeline-watch'
    result = subprocess.run([str(SLURM / 'squeue'), '-h', '-u', 'research_user', '-n', name, '-o', '%i|%T'],
                            capture_output=True, text=True, check=True, timeout=25)
    if result.stdout.strip():
        return
    body = folder / 'supervisor.sbatch'
    body.write_text('#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true WANDB_MODE=disabled DO_NOT_TRACK=1\nexec /usr/bin/python3 ' + str(ROOT / 'pipeline_supervisor.py') + '\n')
    result = subprocess.run([str(SLURM / 'sbatch'), '--parsable', '--partition=defq', '--job-name=' + name,
                             '--cpus-per-task=1', '--mem=8G', '--time=2-00:00:00', '--chdir=' + str(ROOT),
                             '--output=' + str(folder / 'supervisor.private.log'), str(body)],
                            capture_output=True, text=True, env=environment(), check=True, timeout=30)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('pipeline_supervisor_submission_failed')
    save(folder / 'submission.json', dict(state='submitted', job_id=job, submitted_at_utc=now()))


if __name__ == '__main__':
    ensure()
