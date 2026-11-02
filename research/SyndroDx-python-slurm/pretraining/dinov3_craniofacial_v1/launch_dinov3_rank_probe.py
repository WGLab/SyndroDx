"""Probe only idle, newly reserved GPUs; never interrupt processing allocations."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

ROOT = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
POOL = Path('/opt/craniofacial/SyndroDx-python-slurm/pretrain_resources/20261014/single_gpu_expansion/allocations.json')
SLURM = Path('/opt/site/slurm/bin')
os.umask(0o077)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport', choices=['default', 'cumem', 'shared_memory'], default='cumem')
    args = parser.parse_args()
    output = ROOT / ('hardware_probes/single_allocation_group_' + args.transport)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (output / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'completion.json').exists():
            print(json.dumps({'state': 'already_probed', 'output': str(output)})); return
        pool = json.loads(POOL.read_text())
        ids = [str(row['allocation_job_id']) for row in pool['allocations'] if row.get('allocation_job_id')]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise RuntimeError('four_distinct_reservations_required')
        crop_run = Path(Path('/tmp/local_face_label_setup/active_run.txt').read_text().strip())
        crop_allocations = json.loads((crop_run / 'crop_allocations.json').read_text())['allocations']
        crop_ids = {str(v.get('allocation_job_id')) for v in crop_allocations}
        if set(ids) & crop_ids:
            raise RuntimeError('probe_cannot_use_active_crop_allocations')
        query = subprocess.run([str(SLURM / 'squeue'), '--jobs', ','.join(ids), '--noheader',
                                '--format', '%i|%T|%N'], capture_output=True, text=True, timeout=15)
        if query.returncode:
            raise RuntimeError('allocation_query_failed')
        jobs = {v[0]: v[1:] for line in query.stdout.splitlines()
                if len(v := line.strip().split('|')) == 3}
        if any(job not in jobs or jobs[job][0] != 'RUNNING' for job in ids):
            save(output / 'status.json', {'state': 'waiting_for_new_allocations'})
            print(json.dumps({'state': 'waiting_for_new_allocations'})); return
        steps = subprocess.run([str(SLURM / 'squeue'), '--steps', '--jobs', ','.join(ids),
                                '--noheader', '--format', '%i'], capture_output=True, text=True, timeout=15)
        if steps.returncode or any(v.strip() and not v.strip().endswith('.extern')
                                   for v in steps.stdout.splitlines()):
            raise RuntimeError('new_allocation_already_has_active_workload')
        master = jobs[ids[0]][1]
        port = 35000 + secrets.randbelow(20000)
        environment = os.environ.copy()
        environment.update(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           WANDB_DISABLED='true', WANDB_MODE='disabled', NCCL_DEBUG='WARN',
                           TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        if args.transport == 'cumem':
            environment.update(NCCL_CUMEM_ENABLE='1')
        elif args.transport == 'shared_memory':
            environment.update(NCCL_P2P_DISABLE='1', NCCL_CUMEM_HOST_ENABLE='0', NCCL_NVLS_ENABLE='0')
        environment['NCCL_DEBUG'] = 'INFO'
        processes = []
        try:
            for rank, job in enumerate(ids):
                command = [str(SLURM / 'srun'), '--jobid=' + job, '--nodes=1', '--ntasks=1',
                           '--cpus-per-task=2', '--gres=gpu:1', '--exact', '--exclusive', '--mpi=none',
                           str(ROOT / 'venv/bin/python'), str(ROOT / 'dinov3_rank_probe.py'),
                           '--rank', str(rank), '--world-size', str(len(ids)),
                           '--master-address', master, '--master-port', str(port), '--output', str(output)]
                with (output / f'rank_{rank:02d}.private.log').open('ab') as log:
                    processes.append(subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                      stdout=log, stderr=log, env=environment,
                                                      start_new_session=True))
            save(output / 'status.json', {'state': 'synthetic_communication_probe_running',
                 'allocation_job_ids': ids, 'launcher_pids': [p.pid for p in processes],
                 'clinical_data_accessed': False})
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                codes = [p.poll() for p in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError('communication_probe_failed')
                if all(code == 0 for code in codes):
                    results = [json.loads((output / f'rank_{rank:02d}.json').read_text()) for rank in range(len(ids))]
                    if not all(v.get('state') == 'collectives_passed' for v in results):
                        raise RuntimeError('incomplete_communication_receipts')
                    summary = {'state': 'communication_probe_passed', 'world_size': len(ids),
                               'allocation_job_ids': ids, 'node_count': len({v['host'] for v in results}), 'transport_profile': args.transport,
                               'slowest_elapsed_seconds': max(v['elapsed_seconds'] for v in results),
                               'clinical_data_accessed': False, 'pretraining_started': False,
                               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    save(output / 'completion.json', summary)
                    save(output / 'status.json', summary)
                    print(json.dumps(summary)); return
                time.sleep(1)
            raise RuntimeError('communication_probe_timed_out')
        finally:
            # This terminates only our unfinished test steps, never their salloc parents.
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        output = ROOT / 'hardware_probes/single_allocation_group_last_failure'
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        save(output / 'status.json', {'state': 'attention_required', 'error_type': type(error).__name__,
                                     'reason': str(error) if str(error).replace('_', '').isalnum() else 'probe_failed',
                                     'existing_crop_allocations_untouched': True})
        print(json.dumps({'state': 'attention_required', 'error_type': type(error).__name__}))
        raise SystemExit(1)
