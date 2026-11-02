"""Run paired bounded experiments on existing leases; never auto-restart a decline."""
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
from configuration import WORK, BASE, V1, VENDOR, DIAGNOSTICS, ARMS, WORLD, STEPS, SEED, configuration, protocol
from common import SLURM, PYTHON, environment, load, save, now, notify
from allocation_pool import idle_pool, SchedulerUnavailable
from runtime import claim_jobs
from omegaconf import OmegaConf

def run(arm, phase, pool):
    folder = WORK/arm/phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'ckpt').exists() or (folder/'rank_00_receipt.json').exists():
        raise RuntimeError('existing_experiment_needs_review_not_automatic_resume')
    OmegaConf.save(configuration(arm, phase), folder/'config.yaml')
    clients = []
    try:
        port = 35000+int(os.environ['SLURM_JOB_ID'])%15000
        for rank, allocation in enumerate(pool):
            env = environment()
            env['PYTHONPATH'] = os.pathsep.join(map(str, [WORK, VENDOR, BASE, V1, DIAGNOSTICS]))
            env.update(COMPARISON_ARM=arm, COMPARISON_PHASE=phase, COMPARISON_RANK=str(rank),
                       COMPARISON_WORLD=str(WORLD), COMPARISON_MASTER=pool[0]['node'], COMPARISON_PORT=str(port),
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            command = [str(SLURM/'srun'), '--jobid='+allocation['job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=4', '--mem-per-gpu=68G', '--gpus-per-task=1', '--gpu-bind=single:1',
                       '--cpu-bind=none', '--exact',
                       '--overlap' if allocation.get('step_overlap') else '--exclusive',
                       '--mpi=none', '--time=08:00:00',
                       str(PYTHON), str(WORK/'rank.py'), '--config-file', str(folder/'config.yaml'),
                       '--output-dir', str(folder), '--seed', str(SEED), '--no-resume']
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=stream, start_new_session=True))
        save(WORK/'status.json', dict(state='running', arm=arm, phase=phase, world_size=WORLD,
             maximum_steps=STEPS, automatic_scale_up=False, updated_at_utc=now()))
        deadline = time.monotonic()+7.8*3600
        last_progress = time.monotonic()
        previous_marker = None
        while True:
            codes = [client.poll() for client in clients]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('worker_failure_stop_all_ranks')
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('bounded_phase_deadline')
            markers = list(folder.glob('initialization_rank_*.json'))
            markers += list(folder.glob('ckpt/*/.metadata'))
            markers += list(folder.glob('validation/step_*/report.json'))
            marker = max((p.stat().st_mtime for p in markers), default=None)
            if marker != previous_marker:
                previous_marker, last_progress = marker, time.monotonic()
            if time.monotonic()-last_progress > 45*60:
                raise RuntimeError('no_verified_progress_for_45_minutes')
            time.sleep(5)
        receipts = [load(folder/f'rank_{rank:02d}_receipt.json') for rank in range(WORLD)]
        if not all(receipt.get('success') for receipt in receipts):
            raise RuntimeError('worker_receipts_incomplete')
        if len({(r['state'], r['completed_steps']) for r in receipts}) != 1:
            raise RuntimeError('rank_state_mismatch')
        if phase == 'capacity' and not all(r['peak_reserved_GiB'] < 72 and r['free_GiB'] >= 6 for r in receipts):
            raise RuntimeError('synthetic_capacity_headroom_failed')
        receipt = dict(state=receipts[0]['state'], completed_steps=receipts[0]['completed_steps'],
                       world_size=WORLD, max_reserved_GiB=max(r['peak_reserved_GiB'] for r in receipts),
                       completed_at_utc=now(), test_used=False)
        save(folder/'completion.json', receipt)
        return receipt
    finally:
        for client in clients:
            if client.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(client.pid, signal.SIGTERM)
        for client in clients:
            try:
                client.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(client.pid, signal.SIGKILL)


def main():
    os.umask(0o077)
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((WORK/arm/'training/completion.json').exists() for arm in ARMS):
            raise RuntimeError('completed_or_paused_experiment_no_automatic_restart')
        save(WORK/'protocol.json', dict(protocol(), declared_at_utc=now()))
        deadline = time.monotonic()+8.5*86400
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError('GPU_wait_deadline')
            try:
                pool = idle_pool()
            except SchedulerUnavailable:
                save(WORK/'status.json', dict(state='waiting_for_scheduler_recovery',
                     required_H100s=WORLD, training_started=False, updated_at_utc=now()))
                time.sleep(60)
                continue
            if len(pool) >= WORLD:
                break
            save(WORK/'status.json', dict(state='waiting_for_existing_GPU_requests',
                 running_idle_H100s=len(pool), required_H100s=WORLD,
                 selected_allocations_path=str(WORK/'selected_allocations.json'), updated_at_utc=now()))
            time.sleep(60)
        pool = pool[:WORLD]
        locks = claim_jobs([p['job_id'] for p in pool])
        try:
            verified = idle_pool()
            if {p['job_id'] for p in verified} != {p['job_id'] for p in pool}:
                raise RuntimeError('allocation_changed_before_launch')
            pool = verified
            save(WORK/'active_pool.json', dict(allocations=pool, world_size=WORLD,
                 selected_at_utc=now(), automatic_scale_up=False))
            results = {}
            for arm in ARMS:
                run(arm, 'capacity', pool)  # Includes full model/optimizer checkpoint.
                results[arm] = run(arm, 'training', pool)
                if results[arm]['state'] == 'paused_validation_regression':
                    notify('controlled '+arm+' regression', 'Craniofacial pilot paused after validation decline',
                           'A bounded controlled experiment stopped at a complete model/optimizer checkpoint after validation declined. It will not restart automatically. The independent comparison arm retains the same validation guard. Detailed results remain on HPC.')
            common_steps = sorted(set(p.name for p in (WORK/ARMS[0]/'training/validation').glob('step_*')) &
                                  set(p.name for p in (WORK/ARMS[1]/'training/validation').glob('step_*')))
            matched = []
            for step in common_steps:
                reports = {arm:load(WORK/arm/'training/validation'/step/'report.json') for arm in ARMS}
                if all(r.get('state') == 'complete' for r in reports.values()):
                    matched.append(dict(completed_steps=reports[ARMS[0]]['completed_steps'],
                                        arms={arm:r['metrics'] for arm,r in reports.items()}))
            save(WORK/'comparison_summary.json', dict(state='complete', common_checkpoints=matched,
                 original_metrics=load(WORK/'original_runtime_baseline.json').get('metrics'),
                 protocol=protocol(), test_used=False, completed_at_utc=now()))
            report = dict(state='comparison_complete_pending_review', arms=results,
                          automatic_scale_up=False, regressions_not_restarted=True, test_used=False,
                          completed_at_utc=now())
            save(WORK/'status.json', report)
            notify('controlled comparison complete', 'Craniofacial controlled comparison completed',
                   'The bounded head-initialization comparison has finished or paused at its validation guards. No larger training run has been started. Results and checkpoints remain on HPC for review.')
        finally:
            for held in locks:
                held.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK/'status.json', dict(state='paused_execution_or_integrity_failure',
             error_type=type(error).__name__, automatic_restart=False, updated_at_utc=now()))
        notify('controlled comparison failure', 'Craniofacial controlled comparison requires inspection',
               'The controlled pilot stopped because a local execution, resource or integrity check failed. No automatic restart or scale-up is permitted. Completed checkpoints and original encoders remain preserved on HPC.')
        raise
