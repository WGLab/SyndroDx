"""Reserve additional GPUs without launching training or touching clinical data."""
import argparse
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
ROOT = Path('/opt/craniofacial/SyndroDx-python-slurm/pretrain_resources/20261014')
SLURM = Path('/opt/site/slurm/bin')
SPECS = [
    {'key': 'h100_main_8', 'job_name': 'sslprep-1014-h100x8', 'purpose': 'single_node_main_pretraining',
     'partition': 'gpu-xe9680q', 'gpu_type': 'h100', 'gpus': 8, 'cpus': 48, 'memory': '512G', 'days': 10},
    {'key': 'h100_pilot_1a', 'job_name': 'sslprep-1014-h100a', 'purpose': 'environment_and_training_pilot',
     'partition': 'gpu-xe9680q', 'gpu_type': 'h100', 'gpus': 1, 'cpus': 8, 'memory': '96G', 'days': 10,
     'exclude': 'g-44-03'},
    {'key': 'h100_pilot_1b', 'job_name': 'sslprep-1014-h100b', 'purpose': 'parameter_and_baseline_pilot',
     'partition': 'gpu-xe9680q', 'gpu_type': 'h100', 'gpus': 1, 'cpus': 8, 'memory': '96G', 'days': 10,
     'exclude': 'g-44-03'},
    {'key': 'a100_aux_2', 'job_name': 'sslprep-1014-a100x2', 'purpose': 'auxiliary_pretraining_experiments',
     'partition': 'gpuq', 'gpu_type': 'a100', 'gpus': 2, 'cpus': 12, 'memory': '128G', 'days': 5,
     'node': 'g-10-01', 'expected_GPU_memory_GB': 80, 'memory_evidence': 'previous_local_allocation_verified'},
]


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(data):
    data['updated_at_utc'] = timestamp()
    temporary = ROOT / 'allocations.json.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, ROOT / 'allocations.json')


def queue():
    result = subprocess.run([str(SLURM / 'squeue'), '-u', 'research_user', '--noheader',
                             '--format', '%i|%j|%P|%T|%R|%b'],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('scheduler_status_unavailable')
    rows = {}
    for line in result.stdout.splitlines():
        columns = line.strip().split('|')
        if len(columns) == 6:
            job, name, partition, state, reason, gres = columns
            rows[job] = {'job_id': job, 'job_name': name, 'partition': partition,
                         'scheduler_state': state, 'scheduler_reason_or_node': reason, 'gres': gres}
    return rows


def emit(row):
    print(json.dumps({k: row[k] for k in ('key', 'allocation_job_id', 'gpus', 'gpu_type', 'days',
                                        'state', 'scheduler_state', 'scheduler_reason_or_node',
                                        'requester_pid') if k in row}), flush=True)


def refresh(data):
    jobs = queue()
    for row in data['allocations']:
        job_id = str(row.get('allocation_job_id', ''))
        if job_id in jobs:
            row.update(jobs[job_id])
            row['state'] = 'allocated' if jobs[job_id]['scheduler_state'] == 'RUNNING' else 'queued'
        elif job_id:
            row['state'] = 'not_in_active_queue'
    save(data)


def launch(data, spec, jobs):
    existing = next((row for row in data['allocations'] if row['key'] == spec['key']), None)
    match = next((row for row in jobs.values() if row['job_name'] == spec['job_name'] and
                  row['partition'] == spec['partition']), None)
    if match:
        row = existing if existing is not None else dict(spec)
        row.update(match)
        row['allocation_job_id'] = match['job_id']
        row['state'] = 'allocated' if match['scheduler_state'] == 'RUNNING' else 'queued'
        if existing is None:
            data['allocations'].append(row)
        save(data)
        emit(row)
        return
    if existing and existing.get('state') not in ('request_failed', 'not_in_active_queue'):
        # An ambiguous earlier launch is inspected rather than submitted twice.
        emit(existing)
        return
    row = existing if existing is not None else dict(spec)
    if existing and existing.get('allocation_job_id'):
        row.setdefault('previous_allocation_job_ids', []).append(str(existing['allocation_job_id']))
        row.pop('allocation_job_id', None)
        for field in ('job_id', 'scheduler_state', 'scheduler_reason_or_node', 'gres'):
            row.pop(field, None)
    row.update(spec)
    row.update(state='launching', requested_at_utc=timestamp())
    if existing is None:
        data['allocations'].append(row)
    save(data)
    command = [str(SLURM / 'salloc'), '--job-name=' + spec['job_name'],
               '--partition=' + spec['partition'], '--nodes=1', '--ntasks=1',
               '--cpus-per-task=' + str(spec['cpus']),
               '--gres=gpu:' + spec['gpu_type'] + ':' + str(spec['gpus']),
               '--mem=' + spec['memory'], '--time=' + str(spec['days']) + '-00:00:00']
    if 'node' in spec:
        command.append('--nodelist=' + spec['node'])
    if 'exclude' in spec:
        command.append('--exclude=' + spec['exclude'])
    command.extend(['/usr/bin/sleep', str(spec['days'] * 86400)])
    stdout = ROOT / (spec['key'] + '.stdout.log')
    stderr = ROOT / (spec['key'] + '.stderr.log')
    offset = stderr.stat().st_size if stderr.exists() else 0
    with stdout.open('ab') as out, stderr.open('ab') as err:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=err, start_new_session=True)
    row.update(requester_pid=process.pid, state='awaiting_scheduler_reply', command=command,
               allocation_body='sleep_' + str(spec['days'] * 86400))
    save(data)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        with stderr.open('rb') as log:
            log.seek(offset)
            reply = log.read().decode(errors='replace')
        ids = re.findall(r'(?:Pending job allocation|Granted job allocation) (\d+)', reply)
        if ids:
            row.update(allocation_job_id=ids[-1], state='submitted')
            save(data)
            emit(row)
            return
        if process.poll() is not None:
            row.update(state='request_failed', return_code=process.returncode)
            save(data)
            emit(row)
            return
        time.sleep(0.25)
    save(data)
    emit(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    durable_script = ROOT / 'request_salloc.py'
    if Path(__file__).resolve() != durable_script.resolve():
        shutil.copy2(__file__, durable_script)
    with (ROOT / 'requests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = ROOT / 'allocations.json'
        data = json.loads(registry.read_text()) if registry.exists() else {
            'created_at_utc': timestamp(), 'kind': 'pretraining_resource_reservations',
            'training_started': False, 'clinical_data_accessed': False, 'allocations': []}
        jobs = queue()
        if not args.status_only:
            refresh(data)
            for spec in SPECS:
                launch(data, spec, jobs)
                jobs = queue()
        refresh(data)
        print(json.dumps({'registry': str(registry), 'training_started': False,
                          'requests_recorded': len(data['allocations']),
                          'requested_H100_GPUs': sum(r['gpus'] for r in data['allocations'] if r['gpu_type'] == 'h100'),
                          'requested_A100_GPUs': sum(r['gpus'] for r in data['allocations'] if r['gpu_type'] == 'a100')}), flush=True)
        for row in data['allocations']:
            emit(row)


if __name__ == '__main__':
    main()
