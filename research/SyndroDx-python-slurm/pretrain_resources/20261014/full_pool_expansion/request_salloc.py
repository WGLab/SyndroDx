"""Idempotently expand the H100 pool up to the per-user GPU quota."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

os.umask(0o077)
ROOT = Path('/opt/craniofacial/SyndroDx-python-slurm/pretrain_resources/20261014/full_pool_expansion')
SLURM = Path('/opt/site/slurm/bin')


def save(data):
    data['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = ROOT / 'allocations.json.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, ROOT / 'allocations.json')


def query():
    r = subprocess.run([str(SLURM / 'squeue'), '-u', 'research_user', '--noheader',
                        '--format', '%i|%j|%T|%R'], capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise RuntimeError('scheduler_query_failed')
    return {v[1]: {'allocation_job_id': v[0], 'scheduler_state': v[2], 'node_or_reason': v[3]}
            for line in r.stdout.splitlines() if len(v := line.strip().split('|')) == 4}


def main():
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = ROOT / 'request_salloc.py'
    if Path(__file__).resolve() != destination.resolve():
        shutil.copy2(__file__, destination)
    with (ROOT / 'requests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = ROOT / 'allocations.json'
        data = json.loads(registry.read_text()) if registry.exists() else {
            'kind': 'additional_single_GPU_pool', 'training_started': False,
            'existing_crop_allocations_untouched': True, 'allocations': []}
        for slot in range(10):
            name = f'sslmax-1014-h100-{slot}'
            row = next((v for v in data['allocations'] if v['slot'] == slot), None)
            live = query()
            if name in live:
                if row is None:
                    row = {'slot': slot, 'job_name': name, 'gpus': 1, 'days': 10}
                    data['allocations'].append(row)
                row.update(live[name]); save(data)
                continue
            if row is not None:
                # A cancelled or ambiguous prior request is never resubmitted silently.
                continue
            row = {'slot': slot, 'job_name': name, 'gpus': 1, 'gpu_type': 'h100',
                   'partition': 'gpu-xe9680q', 'days': 10, 'cpus': 6, 'memory': '72G',
                   'purpose': 'distributed_pretraining_and_legitimate_framework_pilots',
                   'state': 'launching', 'requested_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            data['allocations'].append(row); save(data)
            stdout = ROOT / f'slot_{slot:02d}.stdout.log'
            stderr = ROOT / f'slot_{slot:02d}.stderr.log'
            args = [str(SLURM / 'salloc'), '--job-name=' + name, '--partition=gpu-xe9680q',
                    '--nodes=1', '--ntasks=1', '--cpus-per-task=6', '--gres=gpu:h100:1',
                    '--mem=72G', '--time=10-00:00:00', '/usr/bin/sleep', '864000']
            with stdout.open('ab') as out, stderr.open('ab') as err:
                process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                           start_new_session=True)
            row.update(requester_pid=process.pid, state='awaiting_scheduler_reply', command=args); save(data)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                message = stderr.read_text(errors='replace')
                ids = re.findall(r'(?:Pending job allocation|Granted job allocation) (\d+)', message)
                if ids:
                    row.update(allocation_job_id=ids[-1], state='submitted'); break
                if process.poll() is not None:
                    row.update(state='request_failed', exit_code=process.returncode); break
                time.sleep(.25)
            save(data)
            print(json.dumps({k: row[k] for k in ('slot', 'allocation_job_id', 'state', 'gpus', 'days') if k in row}), flush=True)
        live = query()
        for row in data['allocations']:
            if row['job_name'] in live:
                row.update(live[row['job_name']])
                row['state'] = 'allocated' if row['scheduler_state'] == 'RUNNING' else 'queued'
            print(json.dumps({k: row[k] for k in ('slot', 'allocation_job_id', 'state', 'scheduler_state', 'node_or_reason') if k in row}), flush=True)
        save(data)
        print(json.dumps({'further_single_H100_requests': len(data['allocations']),
                          'existing_crop_allocations_untouched': True, 'registry': str(registry)}), flush=True)


if __name__ == '__main__':
    main()
