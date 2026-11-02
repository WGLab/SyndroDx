"""Twenty independent ten-day salloc allocations; never cancel a live request."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SLURM = Path('/opt/site/slurm/bin')
PREFIX = 'h100-hold20-'
TARGET = 20
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED',
            'SPECIAL_EXIT'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.chmod(temp, 0o600)
    temp.replace(path)


def environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SLURM_', 'SALLOC_', 'SBATCH_'))}
    env['SLURM_CONF'] = '/opt/site/slurm/slurm.conf'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def query(name, *args):
    # This coordinator has no cancellation, signal, release, or update operation.
    if name not in ('squeue', 'sacct'):
        raise ValueError('read-only scheduler query required')
    return subprocess.run([str(SLURM / name), *args], env=environment(),
                          text=True, capture_output=True, check=True,
                          timeout=45).stdout


def live_rows():
    rows = {}
    output = query('squeue', '-h', '-u', 'research_user', '-o',
                   '%i|%100j|%T|%N|%S|%e|%l|%R')
    for line in output.splitlines():
        v = [item.strip() for item in line.split('|')]
        if len(v) == 8 and re.fullmatch(re.escape(PREFIX) + r's\d{2}-a\d{4,}', v[1]):
            if v[1] in rows:
                raise RuntimeError('duplicate_allocation_name_requires_inspection')
            rows[v[1]] = dict(job_id=v[0], name=v[1], state=v[2], node=v[3],
                             start=v[4], end=v[5], time_limit=v[6], reason=v[7])
    return rows


def history_rows(names):
    if not names:
        return {}
    rows = {}
    output = query('sacct', '-n', '-X', '-u', 'research_user', '-S', '2026-10-27',
                   '--name=' + ','.join(names), '-o', 'JobIDRaw,JobName%100,State', '-P')
    for line in output.splitlines():
        v = [item.strip() for item in line.split('|')]
        if len(v) >= 3 and v[0].isdigit() and v[1] in names:
            if v[1] in rows and rows[v[1]]['job_id'] != v[0]:
                raise RuntimeError('duplicate_submission_requires_inspection')
            rows[v[1]] = dict(job_id=v[0], name=v[1],
                             state=v[2].split()[0].rstrip('+'))
    return rows


def allocation_args(name):
    return [str(SLURM / 'salloc'), '--no-shell', '--no-bell',
            '--account=hpcusers', '--partition=gpu-xe9680q',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
            '--gres=gpu:h100:1', '--mem=96G', '--time=10-00:00:00',
            '--job-name=' + name,
            '--comment=20-slot independent allocations; retain idle; no task coupling']


def request(name):
    # An independent session keeps pending salloc clients alive after chat/guardian exits.
    # --no-shell makes granted allocations independent of all client processes.
    with (ROOT / 'logs' / (name + '.salloc.log')).open('ab', buffering=0) as out:
        result = subprocess.run(allocation_args(name), env=environment(),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=out)
    save(ROOT / 'requests' / (name + '.exit.json'),
         dict(returncode=result.returncode, finished_at_utc=now()))


def launch(slot, old, registry):
    attempt = old.get('attempt', 0) + 1
    name = PREFIX + f's{slot:02d}-a{attempt:04d}'
    row = dict(slot=slot, attempt=attempt, name=name, submitted_at_utc=now(),
               submitted_epoch=time.time(), state='SUBMITTING')
    # Persist intent before launching; an ambiguous outcome is never blindly retried.
    registry[str(slot)] = row
    save(ROOT / 'registry.json', registry)
    with (ROOT / 'logs' / (name + '.launcher.log')).open('ab', buffering=0) as out:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                  '--request', name], env=environment(),
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                 start_new_session=True)
    row.update(launcher_pid=child.pid, launcher_host=socket.gethostname())
    save(ROOT / 'registry.json', registry)
    return child


def tick(registry, children):
    live = live_rows()
    missing = [v['name'] for v in registry.values() if v['name'] not in live]
    history = history_rows(missing)
    for slot in range(TARGET):
        row = registry.get(str(slot), {})
        name = row.get('name')
        if name in live:
            row.update(live[name])
            continue
        if name in history:
            row.update(history[name])
            if row['state'] not in TERMINAL:
                continue
            # Only a scheduler-confirmed terminal allocation may be replaced.
            save(ROOT / 'archive' / (name + '.json'), row)
        elif row:
            receipt = load(ROOT / 'requests' / (name + '.exit.json'))
            log = ROOT / 'logs' / (name + '.salloc.log')
            matches = re.findall(r'(?:job (\d+) queued|Granted job allocation (\d+))',
                                 log.read_text(errors='replace') if log.exists() else '')
            if matches:
                row['job_id'] = next(value for value in matches[-1] if value)
                row['state'] = 'AWAITING_SCHEDULER_VISIBILITY'
                continue
            if not receipt or receipt.get('returncode') == 0:
                # Pending, still submitting, or ambiguous: preserve the request.
                continue
            row['state'] = 'SUBMISSION_REJECTED'
            if time.time() - row['submitted_epoch'] < 300:
                continue
        children.append(launch(slot, row, registry))
    save(ROOT / 'registry.json', registry)
    children[:] = [child for child in children if child.poll() is None]
    # Refresh after submission so the published snapshot includes newly queued requests.
    live = live_rows()
    for row in registry.values():
        if row['name'] in live:
            row.update(live[row['name']])
    save(ROOT / 'registry.json', registry)
    summary = dict(target_gpus=TARGET, gpu_type='h100', partition='gpu-xe9680q',
                   allocation_time_limit='10-00:00:00', allocation_method='salloc --no-shell',
                   running=sum(v['state'] == 'RUNNING' for v in live.values()),
                   pending=sum(v['state'] == 'PENDING' for v in live.values()),
                   managed_live_requests=len(live),
                   automatically_cancel_jobs=False, training_completion_releases_gpus=False,
                   automatically_replace_terminal_allocations=True,
                   controller_pid=os.getpid(), controller_host=socket.gethostname(),
                   updated_at_utc=now(), jobs=list(registry.values()))
    save(ROOT / 'status.json', summary)
    return summary


def supervise(standby=False):
    os.umask(0o077)
    for folder in ('logs', 'requests', 'archive'):
        (ROOT / folder).mkdir(exist_ok=True)
    with (ROOT / 'supervisor.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not standby:
                    raise
                time.sleep(30)
        registry = load(ROOT / 'registry.json')
        children = []
        while True:
            try:
                tick(registry, children)
            except Exception as error:
                save(ROOT / 'supervisor_error.json',
                     dict(error_type=type(error).__name__, message=str(error)[:1200],
                          updated_at_utc=now(), live_requests_left_untouched=True))
            time.sleep(30)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--request':
        request(sys.argv[2])
    else:
        supervise('--standby' in sys.argv)
