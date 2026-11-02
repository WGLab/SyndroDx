#!/usr/bin/env python3
"""Prepare and run isolated, checkpointed DINOv3 continuation on real GPU work.

No raw research records or private logs are printed. Deployment requires write
access to the existing cluster project. No external notifications are sent.
"""
import argparse
import ast
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import statistics
import subprocess
import sys
import time

BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
PHASE = BASE / 'continuation_20261022_convergence'
SLURM = Path('/opt/site/slurm/bin')
PYTHON = BASE / 'venv/bin/python'
PLAN = {
    'state': 'prepared_not_submitted',
    'source_checkpoint_iteration': 2135,
    'start_iteration': 2136,
    'requested_single_gpu_allocations': 20,
    'minimum_training_world_size': 8,
    'allocation_time_limit': '10-00:00:00',
    'partition': 'gpu-xe9680q',
    'segment_steps': 500,
    'minimum_additional_steps': 1000,
    'maximum_additional_steps_before_review': 4000,
    'continuation_learning_rate': 2e-6,
    'continuation_teacher_momentum': 0.999,
    'continuation_teacher_temperature': 0.07,
    'plateau_window_steps': 250,
    'plateau_windows': 4,
    'plateau_relative_mean_change_limit': 0.0005,
    'plateau_relative_slope_limit': 0.0005,
    'plateau_loss_cv_limit': 0.01,
    'plateau_component_relative_change_limit': 0.005,
    'allocation_assembly_seconds': 60,
    'cancel_unselected_allocations': True,
    'release_allocations_on_exit': True,
    'original_model_preserved': True,
    'raw_logs_remain_local': True,
    'external_notifications_enabled': False,
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_numeric_metrics(path):
    """Parse locally; return only whitelisted numerical training scalars."""
    result = {}
    allowed = ('iteration', 'total_loss', 'dino_global_crops_loss',
               'dino_local_crops_loss', 'ibot_loss', 'koleo_loss')
    with Path(path).open() as source:
        for line in source:
            try:
                raw = json.loads(line)
                item = {key: raw[key] for key in allowed
                        if isinstance(raw.get(key), (int, float))
                        and not isinstance(raw.get(key), bool)}
                if not all(math.isfinite(value) for value in item.values()):
                    raise ValueError('nonfinite_numeric_training_metric')
                if 'iteration' in item and 'total_loss' in item:
                    result[int(item['iteration'])] = item
            except json.JSONDecodeError:
                raise ValueError('incomplete_private_metrics_record') from None
    return [result[key] for key in sorted(result)]


def plateau(rows, completed_steps, plan=PLAN):
    """A training-loss plateau is evidence about optimization, not generalization."""
    minimum = plan['minimum_additional_steps']
    window = plan['plateau_window_steps']
    count = plan['plateau_windows']
    end = plan['start_iteration'] + completed_steps
    if completed_steps < minimum:
        return {'state': 'continue', 'reason': 'minimum_training_not_reached'}
    windows = []
    for left in range(end - window * count, end, window):
        samples = [r for r in rows if left <= r['iteration'] < left + window]
        if len(samples) < max(5, window // 15):
            return {'state': 'review_required', 'reason': 'insufficient_numeric_observations'}
        xs = [r['iteration'] for r in samples]
        ys = [r['total_loss'] for r in samples]
        if not all(math.isfinite(y) for y in ys):
            return {'state': 'review_required', 'reason': 'nonfinite_loss'}
        mean = statistics.mean(ys)
        if mean <= 0:
            return {'state': 'review_required', 'reason': 'unexpected_loss_scale'}
        mx = statistics.mean(xs)
        denominator = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - mean) for x, y in zip(xs, ys)) / max(denominator, 1)
        components = {}
        for key in ('dino_global_crops_loss', 'dino_local_crops_loss', 'ibot_loss', 'koleo_loss'):
            values = [r[key] for r in samples if key in r]
            if len(values) == len(samples):
                components[key] = statistics.mean(values)
        windows.append({'mean_loss': mean, 'relative_slope': slope * window / mean,
                        'loss_cv': statistics.pstdev(ys) / mean, 'components': components})
    deltas = [abs(b['mean_loss'] - a['mean_loss']) / abs(a['mean_loss'])
              for a, b in zip(windows, windows[1:])]
    stable = (max(deltas) < plan['plateau_relative_mean_change_limit']
              and max(abs(w['relative_slope']) for w in windows) < plan['plateau_relative_slope_limit']
              and max(w['loss_cv'] for w in windows) < plan['plateau_loss_cv_limit'])
    required = {'dino_global_crops_loss', 'dino_local_crops_loss', 'ibot_loss', 'koleo_loss'}
    if any(set(w['components']) != required for w in windows):
        return {'state': 'review_required', 'reason': 'missing_loss_component_monitoring'}
    component_changes = []
    for a, b in zip(windows, windows[1:]):
        for key in required:
            previous = a['components'][key]
            component_changes.append(abs(b['components'][key] - previous) / max(abs(previous), 1e-8))
    stable = stable and max(component_changes) < plan['plateau_component_relative_change_limit']
    return {'state': 'training_loss_plateau' if stable else 'continue',
            'completed_additional_steps': completed_steps,
            'windows': windows, 'relative_adjacent_mean_changes': deltas,
            'maximum_component_relative_change': max(component_changes),
            'does_not_establish_generalization_or_exclude_representation_collapse': True}


def make_rank_source(source):
    """Small, fail-closed changes to the already exercised training implementation."""
    replacements = [
        ('from runtime import ROOT, RUN, save, load, now',
         'from runtime import ROOT, RUN, save, load, now\nCONTINUATION_PLAN = load(Path(__file__).parent / "plan.json")'),
        ('        upstream.apply_optim_scheduler(optimizer, lr, wd, last_layer)',
         '        lr = last_layer = CONTINUATION_PLAN["continuation_learning_rate"]\n'
         '        momentum = CONTINUATION_PLAN["continuation_teacher_momentum"]\n'
         '        temp = CONTINUATION_PLAN["continuation_teacher_temperature"]\n'
         '        upstream.apply_optim_scheduler(optimizer, lr, wd, last_layer)'),
        ("    if end >= total:\n        export = output / 'synthetic_encoder' if 'synthetic' in str(cfg.student.arch) else ROOT / 'encoder'",
         "    if end > start:\n        export = output.parent / 'encoder'"),
    ]
    # The deployed file's formatting is normalized once before exact changes.
    source = ast.unparse(ast.parse(source)) + '\n'
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError('training_source_changed_requires_review')
        source = source.replace(before, after)
    before = '        if rank == 0 and (iteration % 10 == 0 or iteration == end - 1):'
    if source.count(before) != 1:
        raise RuntimeError('training_metrics_insertion_point_changed')
    insertion = '''        component_values = {}
        if iteration % 10 == 0 or iteration == end - 1:
            for component in ('dino_global_crops_loss', 'dino_local_crops_loss', 'ibot_loss', 'koleo_loss'):
                if component not in metrics:
                    raise RuntimeError('required_loss_component_missing')
                scalar = torch.as_tensor(metrics[component], device='cuda').detach().float().mean().clone()
                dist.all_reduce(scalar)
                scalar /= world
                if not bool(torch.isfinite(scalar)):
                    raise RuntimeError('nonfinite_loss_component')
                component_values[component] = float(scalar)
'''
    source = source.replace(before, insertion + before)
    before = "            with (output / 'metrics.private.jsonl').open('a') as f:"
    if source.count(before) != 1:
        raise RuntimeError('training_metric_writer_changed')
    source = source.replace(before, '            value.update(component_values)\n' + before)
    ast.parse(source)
    return source


def deploy():
    """No mutation of the original training, encoder, dataset, or evaluation."""
    from omegaconf import OmegaConf
    original = load(BASE / 'training/state.json')
    if original.get('state') != 'complete' or original.get('next_iteration') != PLAN['start_iteration']:
        raise RuntimeError('source_training_state_changed')
    source = BASE / 'training/ckpt' / str(PLAN['source_checkpoint_iteration'])
    if not (source / '.metadata').is_file():
        raise RuntimeError('committed_source_checkpoint_required')
    if PHASE.exists():
        raise RuntimeError('continuation_directory_already_exists')
    # Check same-filesystem checkpoint links and enough space for three full saves.
    checkpoint_bytes = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    usage = os.statvfs(BASE)
    if usage.f_bavail * usage.f_frsize < checkpoint_bytes * 4:
        raise RuntimeError('insufficient_checkpoint_rotation_space')
    PHASE.mkdir()
    os.chmod(PHASE, 0o700)
    target = PHASE / 'training'
    checkpoint = target / 'ckpt' / source.name
    checkpoint.mkdir(parents=True)
    for item in source.rglob('*'):
        destination = checkpoint / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(exist_ok=True)
        elif item.is_file() and not item.is_symlink():
            os.link(item, destination)
        else:
            raise RuntimeError('unexpected_checkpoint_entry_type')
    cfg = OmegaConf.load(BASE / 'training/run_config.yaml')
    cfg.train.output_dir = str(target)
    cfg.checkpointing.period = PLAN['segment_steps']
    cfg.checkpointing.max_to_keep = 3
    OmegaConf.save(cfg, target / 'run_config.yaml')
    previous = load(BASE / 'training/execution_policy.json')
    if previous.get('state') != 'active':
        raise RuntimeError('source_sampler_policy_required')
    from batch_policy import progress
    anchor = progress(previous, PLAN['start_iteration'])
    policy = dict(previous, anchor_iteration=PLAN['start_iteration'], anchor_progress=anchor,
                  total_iterations=PLAN['start_iteration'] + PLAN['maximum_additional_steps_before_review'])
    save(target / 'execution_policy.json', policy)
    save(PHASE / 'source_sampler_policy.json', previous)
    save(PHASE / 'plan.json', dict(PLAN, state='ready', prepared_at_utc=now()))
    rank_source = make_rank_source((BASE / 'train_rank.py').read_text())
    (PHASE / 'train_continuation_rank.py').write_text(rank_source)
    (PHASE / 'continue_pretrain.py').write_text(Path(__file__).read_text())
    save(PHASE / 'provenance.json', {
        'source_training_code_sha256': hashlib.sha256((BASE / 'train_rank.py').read_bytes()).hexdigest(),
        'continuation_training_code_sha256': hashlib.sha256(rank_source.encode()).hexdigest(),
        'source_checkpoint_iteration': PLAN['source_checkpoint_iteration'],
        'source_checkpoint_files_hardlinked_without_modification': True,
        'original_training_and_encoder_untouched': True,
    })


def environment():
    sys.path.insert(0, str(BASE))
    from runtime import environment as training_environment
    env = training_environment()
    env['SLURM_CONF'] = '/opt/site/slurm/slurm.conf'
    return env


def lease():
    """Exit if real-work orchestration disappears; this does not count as GPU work."""
    while True:
        path = PHASE / 'heartbeat.json'
        if not path.exists() or time.time() - path.stat().st_mtime > 180:
            return
        if load(path).get('release'):
            return
        time.sleep(5)


def heartbeat():
    save(PHASE / 'heartbeat.json', {'updated_at_utc': now(), 'release': False})


def query_jobs(prefix):
    result = subprocess.run([str(SLURM / 'squeue'), '-u', 'research_user', '-h', '-o', '%i|%T|%N|%j'],
                            capture_output=True, text=True, timeout=25, check=True, env=environment())
    jobs = []
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) == 4 and fields[0].isdigit() and fields[3].startswith(prefix):
            jobs.append(dict(job_id=fields[0], state=fields[1], node=fields[2], name=fields[3]))
    return jobs


def cancel(jobs):
    ids = sorted({j['job_id'] for j in jobs})
    if ids:
        subprocess.run([str(SLURM / 'scancel'), *ids], capture_output=True, timeout=30,
                       check=True, env=environment())


def distributed_run(jobs, program, arguments, timeout):
    processes = []
    try:
        run_id = secrets.token_hex(6)
        port = 35000 + secrets.randbelow(20000)
        for rank, job in enumerate(jobs):
            command = [str(SLURM / 'srun'), '--jobid=' + job['job_id'], '--nodes=1',
                       '--ntasks=1', '--cpus-per-task=4', '--gpus-per-task=1',
                       '--gpu-bind=single:1', '--mem-per-gpu=48G', '--exact', '--exclusive',
                       '--mpi=none', str(PYTHON), str(BASE / 'rank_entry.py'),
                       '--offset', str(rank), '--world', str(len(jobs)), '--master', jobs[0]['node'],
                       '--port', str(port), '--run-id', run_id, '--', str(program), *arguments]
            with (PHASE / ('allocation_' + job['job_id'] + '.private.log')).open('ab') as log:
                processes.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                  stderr=log, env=environment(), start_new_session=True))
        deadline = time.monotonic() + timeout
        last_check = 0
        started = time.monotonic()
        while time.monotonic() < deadline:
            heartbeat()
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('distributed_rank_failed')
            if all(code == 0 for code in codes):
                return
            if time.monotonic() - last_check >= 30:
                active = {j['job_id'] for j in query_jobs('sslcont-' + load(PHASE / 'launch.json')['token'])
                          if j['state'] == 'RUNNING'}
                if any(j['job_id'] not in active for j in jobs):
                    raise RuntimeError('training_allocation_revoked')
                if program.name == 'train_continuation_rank.py':
                    state_path = PHASE / 'training/state.json'
                    if state_path.exists():
                        age = time.time() - state_path.stat().st_mtime
                        state = load(state_path).get('state')
                        limit = 1800 if state == 'initializing' else 1200
                        if age > limit and time.monotonic() - started > limit:
                            raise RuntimeError('training_progress_stalled_release_allocations')
                last_check = time.monotonic()
            time.sleep(5)
        raise RuntimeError('distributed_work_timeout')
    finally:
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


def coordinate():
    # Read-only release verification happens before requesting any GPUs.
    sys.path.insert(0, str(BASE))
    from runtime import digest
    from batch_policy import progress
    release = load(BASE / 'dataset_release/release.json')
    if release.get('state') != 'ssl_data_ready':
        raise RuntimeError('released_dataset_required')
    if load(BASE / 'hardware_validation_ready.json').get('state') != 'passed':
        raise RuntimeError('hardware_validation_required')
    for manifest in release['manifests'].values():
        if digest(manifest['path']) != manifest['sha256']:
            raise RuntimeError('released_manifest_changed')
    lock = (PHASE / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (PHASE / 'launch.json').exists():
        raise RuntimeError('explicit_recovery_required_before_resubmission')
    token = secrets.token_hex(4)
    prefix = 'sslcont-' + token
    save(PHASE / 'launch.json', {'state': 'requesting_allocations', 'token': token,
                                'requested_gpu_count': 20, 'started_at_utc': now()})
    clients = []
    selected = []
    succeeded = False
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        heartbeat()
        for slot in range(20):
            command = [str(SLURM / 'salloc'), '--job-name=' + prefix + f'-{slot:02d}',
                       '--partition=' + PLAN['partition'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=8', '--gres=gpu:h100:1', '--mem=96G',
                       '--time=' + PLAN['allocation_time_limit'], '--no-bell', '--chdir=' + str(PHASE),
                       str(PYTHON), str(PHASE / 'continue_pretrain.py'), '--lease']
            with (PHASE / f'salloc_{slot:02d}.private.log').open('ab') as log:
                clients.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                 stderr=log, env=environment(), start_new_session=True))
        first_grant = None
        while True:
            heartbeat()
            jobs = query_jobs(prefix)
            running = [j for j in jobs if j['state'] == 'RUNNING']
            save(PHASE / 'allocation_status.json', {
                'state': 'assembling_training_pool', 'job_ids': [j['job_id'] for j in jobs],
                'running_allocations': len(running), 'pending_allocations': len(jobs) - len(running),
                'updated_at_utc': now(),
            })
            if all(p.poll() is not None for p in clients):
                raise RuntimeError('all_allocation_requests_ended')
            if running and first_grant is None:
                first_grant = time.monotonic()
            elapsed = time.monotonic() - first_grant if first_grant is not None else 0
            if len(running) == 20 or (elapsed >= 20 and len(running) >= PLAN['minimum_training_world_size']):
                selected = sorted(running, key=lambda j: (j['node'], int(j['job_id'])))
                # Any additional allocation must be cancelled, rather than sit unused.
                selected_ids = {j['job_id'] for j in selected}
                cancel([j for j in jobs if j['job_id'] not in selected_ids])
                break
            if elapsed >= PLAN['allocation_assembly_seconds']:
                raise RuntimeError('insufficient_simultaneous_allocations_no_idle_pool_retained')
            time.sleep(5)
        world = len(selected)
        save(PHASE / 'launch.json', {'state': 'communication_preflight', 'token': token,
                                    'world_size': world, 'job_ids': [j['job_id'] for j in selected],
                                    'updated_at_utc': now()})
        probe = PHASE / 'pool_probe'
        probe.mkdir()
        distributed_run(selected, BASE / 'pool_rank_probe.py', ['--output', str(probe)], 360)
        if any(load(probe / f'rank_{rank:02d}.json').get('state') != 'passed' for rank in range(world)):
            raise RuntimeError('pool_communication_receipts_incomplete')
        target = PHASE / 'training'
        policy = load(target / 'execution_policy.json')
        anchor = progress(load(PHASE / 'source_sampler_policy.json'), PLAN['start_iteration'])
        policy.update(anchor_iteration=PLAN['start_iteration'], anchor_progress=anchor,
                      global_batch_size=policy['batch_size_per_gpu'] * world, world_size=world)
        save(target / 'execution_policy.json', policy)
        while True:
            heartbeat()
            save(PHASE / 'launch.json', {'state': 'training', 'token': token, 'world_size': world,
                                        'job_ids': [j['job_id'] for j in selected], 'updated_at_utc': now()})
            distributed_run(selected, PHASE / 'train_continuation_rank.py',
                            ['--config', str(target / 'run_config.yaml'), '--segment-steps', str(PLAN['segment_steps'])],
                            timeout=12 * 3600)
            state = load(target / 'state.json')
            if state.get('state') not in ('segment_complete', 'complete'):
                raise RuntimeError('committed_training_segment_receipt_required')
            completed = state['next_iteration'] - PLAN['start_iteration']
            decision = plateau(read_numeric_metrics(target / 'metrics.private.jsonl'), completed)
            save(PHASE / 'loss_plateau_check.json', dict(decision, updated_at_utc=now()))
            if decision['state'] == 'training_loss_plateau':
                save(PHASE / 'completion.json', dict(decision, state='training_loss_plateau',
                     next_iteration=state['next_iteration'], completed_at_utc=now()))
                succeeded = True
                break
            if decision['state'] == 'review_required' or completed >= PLAN['maximum_additional_steps_before_review']:
                save(PHASE / 'completion.json', {'state': 'review_required',
                     'reason': decision.get('reason', 'additional_training_budget_reached_without_loss_plateau'),
                     'completed_additional_steps': completed, 'updated_at_utc': now()})
                break
        save(PHASE / 'launch.json', {'state': 'finished' if succeeded else 'review_required',
                                    'token': token, 'updated_at_utc': now()})
    except BaseException as error:
        save(PHASE / 'launch.json', {'state': 'attention_required', 'token': token,
             'error_type': type(error).__name__, 'updated_at_utc': now()})
        with (PHASE / 'coordinator.private_errors.log').open('a') as log:
            import traceback
            traceback.print_exc(file=log)
        raise
    finally:
        save(PHASE / 'heartbeat.json', {'release': True, 'updated_at_utc': now()})
        with contextlib.suppress(Exception):
            cancel(query_jobs(prefix))
        for process in clients:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in clients:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=20)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--coordinate', action='store_true')
    parser.add_argument('--lease', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.deploy:
        deploy()
        print(json.dumps({'state': 'ready_not_submitted'}))
    elif args.coordinate:
        coordinate()
    elif args.lease:
        lease()
    elif args.status:
        result = {'phase_exists': PHASE.exists()}
        for filename, keys in {
            'launch.json': ('state', 'world_size', 'requested_gpu_count', 'job_ids', 'error_type', 'updated_at_utc'),
            'allocation_status.json': ('state', 'job_ids', 'running_allocations', 'pending_allocations', 'updated_at_utc'),
            'training/state.json': ('state', 'iteration', 'total_loss', 'lr', 'world_size',
                                    'next_iteration', 'total_iterations', 'last_checkpoint_iteration', 'updated_at_utc'),
            'completion.json': ('state', 'completed_additional_steps', 'next_iteration', 'completed_at_utc'),
        }.items():
            record = load(PHASE / filename)
            if record:
                result[filename] = {key: record[key] for key in keys if key in record}
        print(json.dumps(result))
    else:
        generated = make_rank_source((BASE / 'train_rank.py').read_text())
        print(json.dumps({'state': 'prepared_not_submitted', 'plan': PLAN,
                          'rank_source_compiles': bool(compile(generated, '<continuation>', 'exec'))}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never print exception messages: they may contain private paths or records.
        print(json.dumps({'state': 'failed', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
