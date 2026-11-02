"""Two matched eight-GPU arms; guarded validation and bounded resource use."""
import concurrent.futures
import contextlib
import fcntl
import os
import signal
import subprocess
import threading
import time
from configuration import WORK, BASE, V1, VENDOR, DIAGNOSTICS, ARMS, WORLD, STEPS, SEED, configuration, protocol
from common import SLURM, PYTHON, environment, load, save, now, notify, digest
from runtime import claim_jobs
from allocation_pool import idle_pools, SchedulerUnavailable
from omegaconf import OmegaConf


def verify_preflight():
    proof = load(WORK/'preflight.json')
    if proof.get('state') != 'passed' or proof.get('tests_passed', 0) < 16:
        raise RuntimeError('preflight_required')
    for name, expected in proof['tested_file_sha256'].items():
        if digest(WORK/name) != expected:
            raise RuntimeError('code_changed_after_preflight')
    for name, expected in proof.get('external_source_sha256', {}).items():
        if digest(name) != expected:
            raise RuntimeError('dependency_changed_after_preflight')


def audit_inputs(steps):
    count = 0
    for step in range(1, steps+1):
        for rank in range(WORLD):
            values = [load(WORK/arm/'training/input_hashes'/f'rank_{rank:02d}_step_{step:06d}.json').get('sha256')
                      for arm in ARMS]
            if not values[0] or values[0] != values[1]:
                raise RuntimeError('paired_input_common_prefix_mismatch_or_missing')
            count += 1
    return dict(state='passed', common_steps=steps, rank_batches_checked=count)


def stop_clients(clients):
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
            client.wait(timeout=15)


def run_arm(arm, phase, pool, abort):
    folder = WORK/arm/phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'ckpt').exists() or (folder/'rank_00_receipt.json').exists():
        raise RuntimeError('existing_experiment_requires_review')
    OmegaConf.save(configuration(arm, phase), folder/'config.yaml')
    clients = []
    try:
        port = 30000+(int(os.environ['SLURM_JOB_ID']) % 10000)*2 + ARMS.index(arm)
        for rank, allocation in enumerate(pool):
            env = environment()
            env['PYTHONPATH'] = os.pathsep.join(map(str, [WORK, VENDOR, BASE, V1, DIAGNOSTICS]))
            env.update(COMPARISON_ARM=arm, COMPARISON_PHASE=phase, COMPARISON_RANK=str(rank),
                       COMPARISON_WORLD=str(WORLD), COMPARISON_MASTER=pool[0]['node'], COMPARISON_PORT=str(port),
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            command = [str(SLURM/'srun'), '--jobid='+allocation['job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=4', f"--mem={allocation['host_memory_gib']}G", '--gpus-per-task=1',
                       '--gpu-bind=single:1', '--cpu-bind=none', '--exact',
                       '--overlap' if allocation['step_overlap'] else '--exclusive', '--mpi=none', '--time=08:00:00',
                       str(PYTHON), str(WORK/'rank.py'), '--config-file', str(folder/'config.yaml'),
                       '--output-dir', str(folder), '--seed', str(SEED), '--no-resume']
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=stream, start_new_session=True))
        save(folder/'execution.json', dict(state='running', arm=arm, phase=phase, world_size=WORLD, updated_at_utc=now()))
        deadline = time.monotonic()+7.8*3600
        progress, previous_marker = time.monotonic(), None
        while True:
            codes = [p.poll() for p in clients]
            if abort.is_set():
                raise RuntimeError('peer_execution_failure')
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('worker_failure')
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('bounded_execution_deadline')
            paths = list(folder.glob('initialization_rank_*.json')) + list(folder.glob('ckpt/*/.metadata'))
            paths += list(folder.glob('validation/step_*/report.json'))
            if (folder/'progress.json').exists(): paths.append(folder/'progress.json')
            marker = max((p.stat().st_mtime for p in paths), default=None)
            if marker != previous_marker:
                previous_marker, progress = marker, time.monotonic()
            if time.monotonic()-progress > 30*60:
                raise RuntimeError('no_verified_progress_for_30_minutes')
            time.sleep(5)
        receipts = [load(folder/f'rank_{rank:02d}_receipt.json') for rank in range(WORLD)]
        if not all(r.get('success') for r in receipts) or len({(r.get('state'),r.get('completed_steps')) for r in receipts}) != 1:
            raise RuntimeError('incomplete_or_inconsistent_rank_receipts')
        if phase == 'capacity' and not all(r['peak_reserved_GiB'] < 72 and r['free_GiB'] >= 6 for r in receipts):
            raise RuntimeError('capacity_gpu_headroom_failed')
        telemetry = [load(folder/'telemetry'/f'rank_{rank:02d}.json') for rank in range(WORLD)]
        if phase == 'capacity' and not all(t.get('cgroups') and t['max_host_resident_fraction'] < .85 for t in telemetry):
            raise RuntimeError('capacity_host_headroom_failed')
        receipt = dict(state=receipts[0]['state'], completed_steps=receipts[0]['completed_steps'],
                       world_size=WORLD, max_reserved_GiB=max(r['peak_reserved_GiB'] for r in receipts),
                       completed_at_utc=now(), test_used=False)
        save(folder/'completion.json', receipt)
        save(folder/'execution.json', dict(receipt, arm=arm, phase=phase))
        if phase == 'training':
            notify('paired-eight '+arm+' '+receipt['state'], 'Craniofacial paired experiment: arm finished',
                   'One bounded eight-GPU arm completed or paused at a validation guard. Checkpoints and aggregate reports remain on HPC. There is no automatic scale-up or restart after a decline.')
        return receipt
    except BaseException as error:
        abort.set()
        save(folder/'execution.json', dict(state='failed', error_type=type(error).__name__, updated_at_utc=now()))
        raise
    finally:
        stop_clients(clients)


def main():
    os.umask(0o077)
    verify_preflight()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline = time.monotonic()+30*60
        while True:
            try:
                pools = idle_pools()
                if all(len(pools[a]) == WORLD for a in ARMS): break
            except SchedulerUnavailable:
                pass
            if time.monotonic() > deadline: raise RuntimeError('allocation_readiness_timeout')
            save(WORK/'status.json', dict(state='waiting_for_selected_allocations', updated_at_utc=now()))
            time.sleep(30)
        locks = claim_jobs([p['job_id'] for arm in ARMS for p in pools[arm]])
        try:
            verified = idle_pools()
            if verified != pools: raise RuntimeError('allocation_changed_before_launch')
            save(WORK/'active_pools.json', dict(arms=pools, selected_at_utc=now()))
            results = {}
            for phase in ('capacity', 'training'):
                save(WORK/'status.json', dict(state='running', phase=phase, active_gpus=16,
                     world_size_per_arm=WORLD, automatic_scale_up=False, updated_at_utc=now()))
                abort = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {arm:executor.submit(run_arm, arm, phase, pools[arm], abort) for arm in ARMS}
                    results[phase] = {arm:futures[arm].result() for arm in ARMS}
                if phase == 'capacity':
                    notify('paired-eight capacity passed', 'Craniofacial paired experiment: capacity checks passed',
                           'Both eight-GPU arms passed synthetic forward/backward, optimizer checkpoint and memory headroom checks. The bounded real-data comparison is starting locally on HPC.')
            common = sorted(set(p.name for p in (WORK/ARMS[0]/'training/validation').glob('step_*')) &
                            set(p.name for p in (WORK/ARMS[1]/'training/validation').glob('step_*')))
            matched = []
            for name in common:
                reports = {arm:load(WORK/arm/'training/validation'/name/'report.json') for arm in ARMS}
                if all(r.get('state') == 'complete' for r in reports.values()):
                    matched.append(dict(completed_steps=reports[ARMS[0]]['completed_steps'],
                                        arms={a:r['metrics'] for a,r in reports.items()}))
            if not matched: raise RuntimeError('no_common_validation')
            audit = audit_inputs(max(r['completed_steps'] for r in matched))
            save(WORK/'comparison_summary.json', dict(state='complete', common_checkpoints=matched,
                 input_audit=audit, original_metrics=load(WORK/'original_runtime_baseline.json')['metrics'],
                 protocol=protocol(), test_used=False, completed_at_utc=now()))
            save(WORK/'status.json', dict(state='comparison_complete_pending_review', results=results,
                 input_audit=audit, automatic_scale_up=False, completed_at_utc=now()))
            notify('paired-eight comparison complete', 'Craniofacial paired comparison completed',
                   'Both bounded arms completed or stopped at their validation guards. Common-step comparisons and paired-input integrity checks are saved on HPC. No larger experiment has been started automatically.')
        finally:
            for held in locks: held.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK/'status.json', dict(state='paused_execution_or_integrity_failure',
             error_type=type(error).__name__, automatic_restart=False, updated_at_utc=now()))
        notify('paired-eight execution failure', 'Craniofacial paired comparison requires inspection',
               'The comparison stopped on a local execution, integrity or memory check. Completed checkpoints are preserved. No automatic retry or expansion is enabled.')
        raise

