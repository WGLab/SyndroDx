"""Idempotent submission after reviewed code and synthetic failure-path tests."""
import fcntl
import subprocess
from configuration import WORK, protocol
from controller import verify_preflight
from common import SLURM, environment, load, save, now


def main():
    with (WORK/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify_preflight()
        if load(WORK/'submission.json').get('job_id'):
            raise RuntimeError('existing_submission_requires_review')
        output=subprocess.run([str(SLURM/'squeue'),'-h','-u','research_user','-n','cranio-paired-eight','-o','%i'],
            env=environment(),capture_output=True,text=True,check=True,timeout=30)
        if output.stdout.strip():raise RuntimeError('existing_controller_not_resubmitted')
        save(WORK/'protocol.json',dict(protocol(),declared_at_utc=now()))
        result=subprocess.run([str(SLURM/'sbatch'),'--parsable',str(WORK/'controller.sbatch')],
            env=environment(),capture_output=True,text=True,timeout=45)
        if result.returncode:raise RuntimeError('controller_submission_failed')
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('invalid_submission_receipt')
        receipt=dict(state='submitted',job_id=job,arms=2,gpus_per_arm=8,total_gpus=16,
                     maximum_steps_per_arm=200,validation_every_steps=25,submitted_at_utc=now())
        save(WORK/'submission.json',receipt)
        print('Submitted paired experiment controller '+job+'; two fixed eight-GPU arms.')


if __name__=='__main__':main()
