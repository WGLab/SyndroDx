"""Benchmark idle allocations only, then resume the committed training checkpoint."""
import contextlib
import os
import secrets
import signal
import statistics
import subprocess
import time
from runtime import ROOT, SLURM, load, save, environment, now, notify
from batch_policy import rebase, select_batch


def heartbeat():
    path = ROOT / 'training/controller_launch.json'
    control = load(path)
    save(path, dict(control, launched_at=time.time()))


def latest_iteration(output):
    committed = [int(p.name) for p in (output / 'ckpt').iterdir()
                 if p.is_dir() and p.name.isdigit() and (p / '.metadata').is_file()]
    if not committed:
        raise RuntimeError('committed_checkpoint_required_for_batch_change')
    return max(committed)


def benchmark(selected, cfg_path, batch, output):
    output.mkdir(parents=True, exist_ok=True)
    world = sum(v['gpus'] for v in selected)
    master = selected[0]['node']
    port = 35000 + secrets.randbelow(20000)
    run_id = 'batch-benchmark-' + secrets.token_hex(6)
    processes = []
    try:
        offset = 0
        for job in selected:
            command = [str(SLURM / 'srun'), '--jobid=' + job['job_id'], '--nodes=1',
                       '--ntasks=' + str(job['gpus']), '--cpus-per-task=4', '--gpus-per-task=1',
                       '--gpu-bind=single:1', '--mem-per-gpu=48G', '--exact', '--exclusive', '--mpi=none',
                       str(ROOT / 'venv/bin/python'), str(ROOT / 'rank_entry.py'), '--offset', str(offset),
                       '--world', str(world), '--master', master, '--port', str(port), '--run-id', run_id,
                       '--', str(ROOT / 'batch_benchmark_rank.py'), '--config', str(cfg_path),
                       '--batch', str(batch), '--output', str(output)]
            with (output / ('allocation_' + job['job_id'] + '.private.log')).open('ab') as log:
                processes.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                  env=environment(), start_new_session=True))
            offset += job['gpus']
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            heartbeat()
            codes = [p.poll() for p in processes]
            if any(c not in (None, 0) for c in codes):
                return dict(state='failed', batch=batch, reason='synthetic_rank_failed')
            if all(c == 0 for c in codes):
                receipts = [load(output / f'rank_{i:02d}.json') for i in range(world)]
                if any(v.get('state') != 'passed' or v.get('world_size') != world
                       or v.get('batch') != batch for v in receipts):
                    return dict(state='failed', batch=batch, reason='incomplete_receipts')
                # Rank times are already the maximum across the whole pool.
                return dict(state='passed', batch=batch, all_ranks_passed=True, world_size=world,
                            median_step_seconds=max(statistics.median(v['step_seconds']) for v in receipts),
                            peak_reserved_GiB=max(v['peak_reserved_GiB'] for v in receipts),
                            clinical_data_accessed=False)
            time.sleep(3)
        return dict(state='failed', batch=batch, reason='benchmark_timeout')
    finally:
        # Only our benchmark clients/steps; keep parent sallocs and checkpoints.
        for process in processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)


def prepare(selected, cfg_path):
    from omegaconf import OmegaConf
    output = ROOT / 'training'
    request_path = output / 'batch_tuning_request.json'
    request = load(request_path)
    policy_path = output / 'execution_policy.json'
    previous = load(policy_path)
    if request.get('state') not in ['pending', 'benchmarking', 'selected'] and not previous:
        return
    start = latest_iteration(output) + 1
    cfg = OmegaConf.load(cfg_path)
    baseline = int(cfg.train.batch_size_per_gpu)
    world = sum(v['gpus'] for v in selected)
    chosen = baseline
    if request.get('state') in ['pending', 'benchmarking']:
        # A resumed launcher never reuses incomplete/old benchmark receipts.
        folder = ROOT / 'hardware_probes' / ('batch_tuning_' + secrets.token_hex(6))
        save(request_path, dict(request, state='benchmarking', checkpoint_iteration=start - 1,
                                benchmark_started_at_utc=now()))
        heartbeat()
        save(output / 'launcher.json', dict(state='running', phase='synthetic_batch_benchmark',
             allocation_job_ids=[v['job_id'] for v in selected], world_size=world, started_at_utc=now()))
        save(output / 'state.json', dict(state='initializing', phase='synthetic_batch_benchmark',
             world_size=world, updated_at_utc=now()))
        results = []
        candidates = request.get('candidate_batches_per_gpu', [int(request.get('candidate_batch_size_per_gpu', 96))])
        candidates = [baseline] + sorted({int(b) for b in candidates if int(b) > baseline})
        for batch in candidates:
            results.append(benchmark(selected, cfg_path, batch, folder / ('batch_' + str(batch))))
            # Stop growth at the first failed or memory-bound configuration.
            if results[-1]['state'] != 'passed' or results[-1].get('peak_reserved_GiB', 999) >= 72:
                break
        chosen = select_batch(results, baseline)
        save(folder / 'completion.json', dict(state='complete', selected_batch_size_per_gpu=chosen,
                                              profiles=results, clinical_data_accessed=False, completed_at_utc=now()))
        request = dict(request, state='selected', selected_batch_size_per_gpu=chosen,
                       baseline_batch_size_per_gpu=baseline, profiles=results,
                       checkpoint_iteration=start - 1, benchmark_completed_at_utc=now())
        save(request_path, request)
    elif request.get('state') == 'selected':
        chosen = int(request['selected_batch_size_per_gpu'])
    plan = load(output / 'run_plan.json')
    base_global = previous.get('base_global_batch_size') or (
        int(plan['batch_size_per_gpu']) * int(plan['initial_world_size']))
    base_total = previous.get('base_total_iterations') or int(cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH)
    # A policy is useful even after a world-size or automatic batch fallback.
    policy = rebase(previous, start, chosen, world, base_total, base_global)
    cfg.train.batch_size_per_gpu = chosen
    # Preserve the original scheduler arrays; train_rank indexes by sample exposure.
    save(policy_path, dict(policy, updated_at_utc=now()))
    temporary = cfg_path.with_suffix('.batch.tmp.yaml')
    OmegaConf.save(cfg, temporary)
    temporary.replace(cfg_path)
    save(request_path, dict(request, state='complete', applied_batch_size_per_gpu=chosen,
                            total_iterations=policy['total_iterations'], applied_at_utc=now(),
                            sample_budget_preserved=True, clinical_data_accessed=False))
    notify('SSL batch profiling completed', 'SSL batch profiling completed',
           'The local synthetic throughput comparison has completed. Training will resume from the committed checkpoint using the selected batch. The original sample budget and patient split remain unchanged; private data and diagnostics remain on HPC.')
