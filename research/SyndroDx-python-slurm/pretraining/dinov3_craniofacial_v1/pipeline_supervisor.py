"""Minute-level local supervision through encoder evaluation, with bounded recovery."""
import concurrent.futures
import datetime
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from runtime import ROOT, RUN, SLURM, load, save, now, scheduler, environment, claim_jobs, notify

FOLDER = ROOT / 'pipeline'


def age(value):
    try:
        return time.time() - datetime.datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0


def telemetry(allocations):
    def probe(job):
        command = [str(SLURM / 'srun'), '--jobid=' + job['job_id'], '--nodes=1', '--ntasks=1',
                   '--cpus-per-task=1', '--cpu-bind=none', '--gres=gpu:1', '--overlap', '--exact', '--mpi=none',
                   '/usr/bin/python3', str(ROOT / 'gpu_pipeline_probe.py')]
        try:
            result = subprocess.run(command, capture_output=True, text=True, env=environment(), timeout=25, check=True)
            return dict(job_id=job['job_id'], **json.loads(result.stdout.strip()))
        except Exception:
            return dict(job_id=job['job_id'], state='probe_unavailable')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(allocations))) as pool:
        probes = list(pool.map(probe, allocations))
    save(FOLDER / 'gpu_telemetry.json', dict(state='sampled', allocations=probes, checked_at_utc=now()))


def checkpoint_writing():
    files = []
    for directory in (ROOT / 'training/ckpt').iterdir():
        if directory.is_dir() and not directory.name.isdigit():
            files.extend(p for p in directory.iterdir() if p.is_file())
    snapshot = dict(bytes=sum(p.stat().st_size for p in files),
                    latest_write=max([p.stat().st_mtime for p in files], default=0))
    previous = load(FOLDER / 'checkpoint_io.json')
    changed = snapshot.get('bytes') != previous.get('bytes') or snapshot['latest_write'] != previous.get('latest_write')
    save(FOLDER / 'checkpoint_io.json', dict(snapshot, checked_at_utc=now()))
    return bool(files) and (changed or time.time() - snapshot['latest_write'] < 600)


def training_monitor(jobs, busy):
    state = load(ROOT / 'training/state.json')
    launcher = load(ROOT / 'training/launcher.json')
    if launcher.get('state') == 'running':
        save(FOLDER / 'last_training_attempt.json', launcher)
    if launcher.get('state') == 'running' and state.get('state') == 'training' and age(state.get('updated_at_utc')) > 900:
        if checkpoint_writing():
            return 'checkpoint_io_progress'
        notify('SSL monitored slow progress ' + launcher.get('started_at_utc', ''), 'SSL progress requires attention',
               'The local monitor detected stalled training progress. It is checking the committed checkpoint and project-owned training clients. GPU reservations and all private data remain on HPC.')
        if age(state.get('updated_at_utc')) > 1800:
            history = load(FOLDER / 'recovery_history.json', {'attempts': []})
            recent = [a for a in history['attempts'] if age(a['at_utc']) < 86400]
            if len(recent) >= 3 or any(a['attempt'] == launcher.get('started_at_utc') for a in recent):
                return 'stalled_recovery_limit_or_already_requested'
            coordinator = load(ROOT / 'coordinator_submission.json').get('job_id')
            if not coordinator:
                return 'stalled_coordinator_unknown'
            # srun enforces the CPU coordinator allocation. Never issue scancel
            # against a parent allocation or kill an unverified/user process.
            command = [str(SLURM / 'srun'), '--jobid=' + str(coordinator), '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=1', '--mem=1G', '--gres=none', '--cpu-bind=none', '--overlap', '--exact', '--mpi=none',
                       '/usr/bin/python3', str(ROOT / 'recover_stalled_training.py'), '--expected-start', launcher['started_at_utc']]
            with (FOLDER / 'recovery.private.log').open('ab') as log:
                result = subprocess.run(command, env=environment(), stdout=log, stderr=log, timeout=120)
            recent.append(dict(attempt=launcher['started_at_utc'], at_utc=now(), returncode=result.returncode))
            save(FOLDER / 'recovery_history.json', dict(attempts=recent))
            return 'bounded_stall_recovery_requested'
        return 'progress_stale'
    if launcher.get('state') == 'attention_required':
        control = load(ROOT / 'training/controller_launch.json')
        active_ids = {k for k, v in jobs.items() if v['state'] == 'RUNNING'}
        if not (active_ids & busy) and control.get('failures', 0) > 3:
            history = load(FOLDER / 'retry_history.json', {'attempts': []})
            recent = [a for a in history['attempts'] if age(a['at_utc']) < 86400]
            if len(recent) < 2:
                save(ROOT / 'training/controller_launch.json', dict(control, failures=0, launched_at=0))
                recent.append(dict(at_utc=now(), reason='idle_failed_training_retry_budget'))
                save(FOLDER / 'retry_history.json', dict(attempts=recent))
                return 'bounded_automatic_retry_enabled'
            notify('SSL recovery budget exhausted', 'SSL recovery requires intervention',
                   'Bounded local recovery attempts were exhausted. Existing reservations and checkpoints are retained. The pipeline will not claim completion; private failure diagnostics remain on HPC.')
    return 'monitoring_training'


def run_evaluation_workers(active, busy):
    from prepare_evaluation import prepare_inputs
    folder = ROOT / 'evaluation'
    plan = load(folder / 'input_plan.json')
    if not plan:
        if any(job['job_id'] in busy for job in active):
            return 'waiting_for_training_steps_to_exit'
        if not active:
            return 'waiting_for_evaluation_GPUs'
        plan = prepare_inputs(len(active))
    registry = load(folder / 'launches.json')
    idle = [job for job in active if job['job_id'] not in busy]
    for slot in range(plan['world_size']):
        destination = folder / 'features' / f'slot_{slot:02d}'
        completion = load(destination / 'completion.json')
        if completion.get('state') == 'complete' and completion.get('inputs_sha256') == plan['inputs_sha256']:
            continue
        previous = registry.get(str(slot), {})
        if previous.get('job_id') in busy or time.time() - previous.get('launched_at', 0) < 180:
            continue
        if previous.get('attempts', 0) >= 3:
            notify('SSL evaluation worker failure ' + str(slot), 'Encoder evaluation requires attention',
                   'An offline encoder-evaluation worker could not complete after automatic retries. Training checkpoints and GPU reservations remain intact. Evaluation is incomplete; private diagnostics remain on HPC.')
            continue
        if not idle:
            break
        job = idle.pop(0)
        locks = claim_jobs([job['job_id']])
        try:
            command = [str(SLURM / 'srun'), '--jobid=' + job['job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=4', '--gres=gpu:1', '--mem-per-gpu=48G', '--exact', '--exclusive', '--mpi=none',
                       str(ROOT / 'venv/bin/python'), str(ROOT / 'evaluation_worker.py'), '--slot', str(slot),
                       '--slots', str(plan['world_size'])]
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / 'worker.private.log').open('ab') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                           env=environment(), start_new_session=True)
            registry[str(slot)] = dict(job_id=job['job_id'], pid=process.pid, launched_at=time.time(), attempts=previous.get('attempts', 0) + 1)
            save(folder / 'launches.json', registry)
            busy.add(job['job_id'])
        finally:
            for lock in locks:
                lock.close()
    complete = all(load(folder / 'features' / f'slot_{i:02d}' / 'completion.json').get('state') == 'complete'
                   for i in range(plan['world_size']))
    return 'features_complete' if complete else 'offline_baseline_and_encoder_feature_extraction'


def tick():
    jobs, busy = scheduler()
    active = sorted([v for v in jobs.values() if v['state'] == 'RUNNING' and v['gpus'] == 1], key=lambda v: int(v['job_id']))
    previous = load(FOLDER / 'gpu_telemetry.json')
    if age(previous.get('checked_at_utc')) > 300 or not previous:
        telemetry(active)
    done = load(ROOT / 'evaluation/completion.json')
    if done.get('state') == 'evaluation_complete':
        notify('Craniofacial encoder evaluation completed', 'Encoder training and evaluation completed',
               'Training, encoder export and the offline baseline-controlled evaluation have completed. The local report distinguishes metadata-label results from clinical validation. All images, identifiers, embeddings and model weights remain on HPC. Review the local evaluation report before drawing conclusions.')
        save(RUN / 'pipeline_evaluation_completion.json', dict(state='complete', completed_at_utc=now()))
        return 'complete'
    if load(ROOT / 'training/state.json').get('state') != 'complete' or not (ROOT / 'encoder/encoder_export.json').is_file():
        return training_monitor(jobs, busy)
    state = run_evaluation_workers(active, busy)
    if state != 'features_complete':
        return state
    if load(ROOT / 'evaluation/labels_status.json').get('state') != 'exploratory_metadata_labels_ready':
        notify('SSL label bridge not ready', 'Encoder evaluation label bridge requires attention',
               'Local feature extraction completed, but the authorized metadata-label join has not passed validation. No successful diagnosis evaluation is claimed. The mapping, identifiers and private diagnostics remain on HPC.')
        return 'awaiting_verified_metadata_label_bridge'
    control = load(ROOT / 'evaluation/metrics_launch.json')
    if control.get('job_id'):
        query = subprocess.run([str(SLURM / 'squeue'), '-h', '-j', control['job_id'], '-o', '%T'], capture_output=True, text=True, timeout=25)
        if query.returncode == 0 and query.stdout.strip():
            return 'computing_patient_level_comparison'
    if control.get('attempts', 0) >= 3:
        notify('SSL evaluation metrics failed', 'Encoder evaluation requires attention',
               'Feature extraction completed, but local evaluation metrics did not complete after automatic retries. No successful evaluation is claimed. Private results and diagnostics remain on HPC.')
        return 'evaluation_metrics_require_attention'
    body = ROOT / 'evaluation/metrics.sbatch'
    body.write_text('#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true WANDB_MODE=disabled DO_NOT_TRACK=1\nexec ' + str(ROOT / 'venv/bin/python') + ' ' + str(ROOT / 'evaluate_encoder.py') + '\n')
    command = [str(SLURM / 'sbatch'), '--parsable', '--partition=defq', '--job-name=ssl-offline-evaluation',
               '--cpus-per-task=4', '--mem=24G', '--time=1-00:00:00', '--chdir=' + str(ROOT),
               '--output=' + str(ROOT / 'evaluation/metrics.private.log'), str(body)]
    result = subprocess.run(command, env=environment(), capture_output=True, text=True, check=True, timeout=30)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('evaluation_metrics_submission_failed')
    save(ROOT / 'evaluation/metrics_launch.json', dict(job_id=job, attempts=control.get('attempts', 0) + 1, submitted_at_utc=now()))
    return 'patient_level_evaluation_submitted'


def main():
    os.umask(0o077)
    FOLDER.mkdir(exist_ok=True)
    lock = (FOLDER / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    while True:
        try:
            state = tick()
            save(FOLDER / 'status.json', dict(state=state, checked_at_utc=now()))
            if state == 'complete':
                return
        except Exception as exc:
            save(FOLDER / 'status.json', dict(state='recovering_monitor', error_type=type(exc).__name__, checked_at_utc=now()))
        time.sleep(60)


if __name__ == '__main__':
    main()
