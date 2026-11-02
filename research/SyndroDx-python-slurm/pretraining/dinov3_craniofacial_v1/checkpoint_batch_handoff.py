"""At a newly committed checkpoint, restart only this project training group."""
import argparse
import contextlib
import fcntl
import os
import signal
import subprocess
import time
from runtime import ROOT, SLURM, load, save, now, environment, notify
from recover_stalled_training import verified


REQUEST = ROOT / 'training/checkpoint_batch_handoff.json'


def committed(output, iteration, world):
    folder = output / 'ckpt' / str(iteration)
    metadata = folder / '.metadata'
    if not metadata.is_file() or metadata.stat().st_size == 0:
        return False
    shards = list(folder.glob('*.distcp'))
    return len(shards) >= world and all(p.stat().st_size > 0 for p in shards)


def apply():
    with (ROOT / 'training/checkpoint_batch_apply.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return apply_locked()


def apply_locked():
    request = load(REQUEST)
    if request.get('state') not in ['waiting_for_checkpoint', 'applying']:
        return
    output = ROOT / 'training'
    target, expected = int(request['target_checkpoint_iteration']), request['expected_started_at_utc']
    launcher = load(output / 'launcher.json')
    world = int(launcher.get('world_size', 0))
    if launcher.get('state') != 'running' or launcher.get('started_at_utc') != expected or launcher.get('phase'):
        save(REQUEST, dict(request, state='superseded_by_other_training_attempt', updated_at_utc=now()))
        return
    if not committed(output, target, world):
        raise RuntimeError('target_checkpoint_not_committed')
    clients = [int(p) for p in launcher.get('launcher_pids', [])]
    jobs = [str(v) for v in launcher.get('allocation_job_ids', [])]
    control = load(output / 'controller_launch.json')
    launcher_pid = int(control.get('pid', 0))
    if len(clients) != len(jobs) or not clients or not all(verified(pid, 'train_rank.py', jobs) for pid in clients) or not verified(launcher_pid, 'launch_training.py'):
        raise RuntimeError('project_training_clients_not_all_verified_on_coordinator')
    # Fence the existing controller's retry/startup branch while the old
    # launcher notices its clients exit and releases its resource locks.
    save(output / 'controller_launch.json', dict(control, launched_at=time.time()))
    save(REQUEST, dict(request, state='applying', updated_at_utc=now()))
    for pid in clients:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if not any(verified(p, 'train_rank.py', jobs) for p in clients) and not verified(launcher_pid, 'launch_training.py'):
            break
        time.sleep(1)
    else:
        # Leave normal failure recovery active. Never mark a running group as
        # segment_complete and never touch the parent salloc reservations.
        raise RuntimeError('old_training_group_has_not_fully_exited')
    current = load(output / 'launcher.json')
    if current.get('state') == 'running' and current.get('started_at_utc') != expected:
        raise RuntimeError('new_training_attempt_already_started')
    if not committed(output, target, world):
        raise RuntimeError('checkpoint_receipt_changed_before_resume')
    # A planned, checkpoint-backed handoff is a segment boundary, not an OOM.
    previous_state = load(output / 'state.json')
    save(output / 'state.json', dict(state='segment_complete', last_checkpoint_iteration=target,
         next_iteration=target + 1, total_iterations=previous_state.get('total_iterations'),
         world_size=world, architecture='vit_7b', reason='requested_batch_profile_at_next_checkpoint', updated_at_utc=now()))
    save(output / 'launcher.json', dict(state='segment_complete', world_size=world,
         reason='requested_checkpoint_batch_handoff', completed_at_utc=now()))
    fresh = load(output / 'controller_launch.json')
    save(output / 'controller_launch.json', dict(fresh, launched_at=0, failures=control.get('failures', 0)))
    save(REQUEST, dict(request, state='restart_enabled', checkpoint_iteration=target,
         next_iteration=target + 1, parent_allocations_cancelled=False, updated_at_utc=now()))
    notify('SSL checkpoint batch handoff completed', 'SSL batch adjustment started',
           'The next training checkpoint is committed. The project training group was handed off for offline synthetic batch profiling, preserving its optimizer, patient split and GPU reservations. Private checkpoints and all data remain on HPC.')


def watch():
    with (ROOT / 'training/checkpoint_batch_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return watch_locked()


def watch_locked():
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        request = load(REQUEST)
        if request.get('state') != 'waiting_for_checkpoint':
            return
        launcher = load(ROOT / 'training/launcher.json')
        if launcher.get('started_at_utc') != request['expected_started_at_utc']:
            save(REQUEST, dict(request, state='superseded_by_other_training_attempt', updated_at_utc=now()))
            return
        world = int(launcher.get('world_size', 0))
        if committed(ROOT / 'training', request['target_checkpoint_iteration'], world):
            command = [str(SLURM / 'srun'), '--jobid=' + request['coordinator_job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=1', '--mem=1G', '--gres=none', '--cpu-bind=none', '--overlap', '--exact', '--mpi=none',
                       '/usr/bin/python3', str(ROOT / 'checkpoint_batch_handoff.py'), '--apply']
            with (ROOT / 'training/checkpoint_batch_handoff.private.log').open('ab') as log:
                result = subprocess.run(command, env=environment(), stdout=log, stderr=log, timeout=150)
            if result.returncode:
                raise RuntimeError('checkpoint_batch_handoff_apply_failed')
            return
        time.sleep(2)
    raise RuntimeError('next_checkpoint_wait_timeout')


def validate():
    request = load(REQUEST)
    launcher = load(ROOT / 'training/launcher.json')
    control = load(ROOT / 'training/controller_launch.json')
    jobs = [str(v) for v in launcher.get('allocation_job_ids', [])]
    clients = [int(v) for v in launcher.get('launcher_pids', [])]
    passed = (launcher.get('state') == 'running'
              and launcher.get('started_at_utc') == request['expected_started_at_utc']
              and bool(clients) and len(clients) == len(jobs)
              and all(verified(p, 'train_rank.py', jobs) for p in clients)
              and verified(int(control.get('pid', 0)), 'launch_training.py'))
    save(ROOT / 'training/checkpoint_batch_preflight.json', dict(state='passed' if passed else 'failed',
         verified_clients=len(clients) if passed else 0, no_processes_stopped=True, updated_at_utc=now()))
    if not passed:
        raise RuntimeError('checkpoint_batch_process_preflight_failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    try:
        validate() if args.validate else apply() if args.apply else watch()
    except Exception as exc:
        request = load(REQUEST)
        save(REQUEST, dict(request, state='requires_attention', error_type=type(exc).__name__,
             reason=str(exc) if str(exc).replace('_', '').isalnum() else 'local_error', updated_at_utc=now()))
        notify('SSL checkpoint batch handoff requires attention', 'SSL batch adjustment requires attention',
               'The planned local checkpoint handoff did not complete validation. GPU reservations and checkpoints are retained, and ordinary training recovery remains enabled. No patient information is included in this message.')
        raise SystemExit(1)
