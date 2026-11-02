"""Stop only the verified stalled project training clients, never allocations."""
import argparse
import contextlib
import os
import signal
import time
from pathlib import Path
from runtime import ROOT, load, save, now


def verified(pid, script, jobs=None):
    try:
        proc = Path('/proc') / str(pid)
        args = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if proc.stat().st_uid != os.getuid() or str(ROOT / script) not in args or os.getpgid(pid) != pid:
            return False
        if jobs is not None:
            return Path(args[0]).name == 'srun' and any('--jobid=' + job in args for job in jobs)
        return True
    except (OSError, UnicodeError):
        return False


def main(expected):
    path = ROOT / 'training/launcher.json'
    launcher = load(path)
    if launcher.get('state') != 'running' or launcher.get('started_at_utc') != expected or launcher.get('phase') == 'synthetic_batch_benchmark':
        return
    state = load(ROOT / 'training/state.json')
    if state.get('state') != 'training':
        return
    clients = [int(pid) for pid in launcher.get('launcher_pids', [])]
    jobs = [str(job) for job in launcher.get('allocation_job_ids', [])]
    stopped = 0
    for pid in clients:
        if verified(pid, 'train_rank.py', jobs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
                stopped += 1
    # The launcher notices a failed client and tears down its own group.
    for _ in range(15):
        if load(path).get('state') != 'running':
            break
        time.sleep(2)
    for pid in clients:
        if verified(pid, 'train_rank.py', jobs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
    current = load(path)
    if current.get('state') == 'running' and current.get('started_at_utc') != expected:
        return
    control = load(ROOT / 'training/controller_launch.json')
    pid = int(control.get('pid', 0))
    if pid > 0 and verified(pid, 'launch_training.py'):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)
    if stopped:
        save(path, dict(state='attention_required', reason='verified_stalled_training_group_stopped',
                        updated_at_utc=now()))
    save(ROOT / 'pipeline/stall_recovery_receipt.json', dict(state='clients_stopped' if stopped else 'no_verified_clients',
         stopped_clients=stopped, allocations_cancelled=False, expected_attempt=expected, updated_at_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-start', required=True)
    main(parser.parse_args().expected_start)
