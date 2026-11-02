"""Idempotent submission after synthetic-only CPU checks."""
import fcntl
import subprocess
from configuration import WORK, WORLD, protocol
from common import SLURM, environment, load, save, now, digest

NAME = 'cranio-controlled-guarded'


def main():
    with (WORK/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proof = load(WORK/'preflight.json')
        if proof.get('state') != 'passed' or proof.get('synthetic_tests_passed') != 9:
            raise RuntimeError('synthetic_preflight_required')
        if proof.get('allocation_pool_tests_passed') != 6:
            raise RuntimeError('allocation_pool_preflight_required')
        for name, expected in proof['tested_file_sha256'].items():
            if digest(WORK/name) != expected:
                raise RuntimeError('code_changed_since_preflight')
        result = subprocess.run([str(SLURM/'squeue'), '-h', '-u', 'research_user', '-n', NAME, '-o', '%i|%T'],
                                env=environment(), capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip():
            print('Existing controlled comparison retained: '+result.stdout.strip())
            return
        if load(WORK/'submission.json').get('job_id'):
            raise RuntimeError('previous_controller_exists_review_before_resubmitting')
        save(WORK/'protocol.json', dict(protocol(), declared_at_utc=now()))
        save(WORK/'status.json', dict(state='prepared_for_submission', automatic_scale_up=False,
                                     updated_at_utc=now()))
        result = subprocess.run([str(SLURM/'sbatch'), '--parsable', str(WORK/'controller.sbatch')],
                                env=environment(), capture_output=True, text=True, timeout=45)
        if result.returncode:
            save(WORK/'submission_failure.json', dict(state='failed', scheduler_error=result.stderr.strip(),
                                                     updated_at_utc=now()))
            raise RuntimeError('scheduler_rejected_controller_submission')
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('invalid_submission_receipt')
        receipt = dict(state='submitted', job_id=job,
                       existing_GPU_selection=str(WORK/'selected_allocations.json'),
                       requested_GPUs=WORLD, bounded_steps_per_arm=200,
                       blocking_validation_every_steps=25, automatic_scale_up=False,
                       submitted_at_utc=now())
        save(WORK/'submission.json', receipt)
        print(receipt)


if __name__ == '__main__':
    main()
