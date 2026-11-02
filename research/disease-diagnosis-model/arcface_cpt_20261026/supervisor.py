"""Adopt existing Slurm leases without cancelling allocations or queued jobs."""
import fcntl
import os
import signal
import socket
import subprocess
import time
from control import WORK, OLD, PYTHON, SLURM, load, save, now, environment, identity, alive

def heartbeat(state, **extra):
    row = dict(state=state, updated_at_utc=now(), **extra)
    save(WORK/'supervisor_status.json', row)
    # Already-running lease.py processes require this heartbeat to stay alive.
    save(OLD/'supervisor_heartbeat.json', dict(row, owner='arcface_cpt_20261026'))

def main():
    os.umask(0o077)
    with (WORK/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_supervisor = load(OLD/'launcher_process.json')
        old_controller = load(OLD/'supervised_controller.json')
        if old_supervisor.get('host') != socket.gethostname():
            raise RuntimeError('old_supervisor_host_mismatch')
        pid = old_supervisor.get('pid')
        if identity(pid):
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read()
            if str(OLD/'launch_pool.py').encode() not in cmd:
                raise RuntimeError('old_supervisor_identity_mismatch')
            # Signal only the coordinator, never its child process group or jobs.
            os.kill(pid, signal.SIGTERM)
        with (OLD/'launcher.lock').open('a') as old_lock:
            fcntl.flock(old_lock, fcntl.LOCK_EX)
            heartbeat('requesting_dino_checkpoint_and_retaining_allocations')
            save(OLD/'stop_requested.json', dict(reason='user_switch_to_arcface_keep_gpu_allocations',
                requested_at_utc=now()))
            save(WORK/'transition.json', dict(state='old_training_checkpoint_requested',
                old_controller=old_controller, kept_registry=str(OLD/'pool_registry.json'),
                cancel_allocations=False, updated_at_utc=now()))
            proc = None
            adopted = load(WORK/'controller_process.json')
            last_reconcile = 0.
            while True:
                if time.monotonic()-last_reconcile>60:
                    try:
                        from renew_leases import reconcile
                        reconcile()
                    except Exception as error:
                        save(WORK/'lease_reconcile_status.json', dict(state='retrying',
                            error_type=type(error).__name__,updated_at_utc=now()))
                    last_reconcile = time.monotonic()
                if alive(old_controller):
                    heartbeat('waiting_for_dino_checkpoint', allocations_retained=True)
                elif not (WORK/'ready.json').exists():
                    heartbeat('preparing_arcface_training', allocations_retained=True,
                        dino_controller_exited=True)
                    save(WORK/'transition.json', dict(state='dino_exited_allocations_retained',
                        old_exit=load(OLD/'controller_exit.json'),
                        checkpoint_boundary=load(OLD/'latest_resize_boundary.json'),
                        updated_at_utc=now()))
                else:
                    running = proc.poll() is None if proc is not None else alive(adopted)
                    if not running:
                        if any((WORK/name).exists() for name in ('training_complete.json','health_stop.json','stop_requested.json')):
                            heartbeat('checkpoint_preserved_review_required', allocations_retained=True)
                            time.sleep(10)
                            continue
                        with (WORK/'logs/controller.private.log').open('ab') as stream:
                            proc = subprocess.Popen([str(PYTHON), str(WORK/'controller.py')],
                                env=environment(), stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=stream, start_new_session=True)
                        save(WORK/'controller_process.json', dict(pid=proc.pid,
                            start_identity=identity(proc.pid), host=socket.gethostname(), updated_at_utc=now()))
                        adopted = load(WORK/'controller_process.json')
                    heartbeat('supervising_arcface_training', controller_pid=adopted.get('pid'),
                        allocations_retained=True)
                time.sleep(10)

if __name__ == '__main__':
    main()
