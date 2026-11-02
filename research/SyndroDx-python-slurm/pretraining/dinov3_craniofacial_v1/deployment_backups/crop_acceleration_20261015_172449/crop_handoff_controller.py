"""Reuse idle Slurm allocations for private, resumable craniofacial cropping."""
import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
SLURM = '/opt/site/slurm/bin/'
TERMINAL_HOLDS = {'held_pilot_acceptance', 'held_repeated_errors', 'failed'}


def load(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if fallback is None else fallback


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scheduler():
    result = subprocess.run([SLURM+'squeue', '-h', '-u', 'research_user', '-o', '%i|%T'],
                            capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError('allocation_queue_unavailable')
    jobs = dict(line.split('|', 1) for line in result.stdout.splitlines() if '|' in line)
    result = subprocess.run([SLURM+'squeue', '-h', '--steps', '-u', 'research_user', '-o', '%i'],
                            capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError('step_queue_unavailable')
    busy = {line.strip().split('.')[0] for line in result.stdout.splitlines()
            if line.strip() and not line.strip().endswith(('.extern', '.batch'))}
    return jobs, busy


def allocations(run):
    items = []
    for name in ('single_gpu_allocations.json', 'a10080_allocations.json', 'crop_allocations.json'):
        items.extend(load(run/name).get('allocations', []))
    return [{'job':str(x['allocation_job_id']), 'replica':int(x['worker'])}
            for x in items if x.get('allocation_job_id')]


def idle_allocations(run, items, jobs, busy, reserved):
    annotation_complete = load(run/'completion.json').get('state') == 'annotation_complete'
    return [x for x in items if jobs.get(x['job']) == 'RUNNING'
            and x['job'] not in busy and x['job'] not in reserved
            and (annotation_complete or load(run/'replicas'/f"slot_{x['replica']:02d}.json").get('state') == 'worker_finished')]


def completed_shards(run):
    return sum(load(p).get('state') == 'completed'
               for p in (run/'direct_workers').glob('shard_*.status.json'))


def ensure_crop_capacity(root, worker_count):
    plan = load(root/'handoff_plan.json')
    if not plan.get('request_remaining_GPUs_after_first_pilot'):
        return
    first_passed = any(load(root/'pilot'/f'slot_{i:02d}.json').get('passed') is True
                       and load(root/'gpu_smoke'/f'slot_{i:02d}.json').get('state') == 'synthetic_gpu_checks_passed'
                       for i in range(worker_count))
    if not first_passed:
        return
    registered = {int(x['worker']) for x in load(root.parent/'crop_allocations.json').get('allocations', [])}
    launched = load(root/'capacity_requests.json')
    for slot in range(worker_count):
        if slot in registered or str(slot) in launched:
            continue
        with (root/'capacity_request.log').open('a') as log:
            process = subprocess.Popen(['/usr/bin/python3', str(root/'request_crop_allocation.py'),
                                        '--slot', str(slot)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        launched[str(slot)] = {'state':'requester_launched', 'pid':process.pid, 'launched_at':time.time()}
        save(root/'capacity_requests.json', launched)
        return  # One request per tick; all allocation identifiers are registered durably.


def notify(root, event, title, message):
    spec = importlib.util.spec_from_file_location('private_stage_notifications', root.parent/'notify_pipeline_stages.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.notify(event, title, message)


def finalize(root, worker_count):
    run = root.parent
    if load(run/'completion.json').get('state') != 'annotation_complete':
        return False
    if not all(load(root/'workers'/f'slot_{i:02d}.json').get('state') == 'finished_provisional'
               for i in range(worker_count)):
        return False
    with (run/'accepted_faces.tsv').open(newline='') as handle:
        accepted = {row['photo_id']:row for row in csv.DictReader(handle, delimiter='\t')}
    records = {}
    extra = 0
    for i in range(worker_count):
        path = root/'records'/f'slot_{i:02d}.sqlite'
        db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise RuntimeError('crop_database_integrity_failed')
            for (encoded,) in db.execute('SELECT record FROM records'):
                value = json.loads(encoded)
                photo = value['photo_id']
                if photo not in accepted:
                    extra += 1
                    continue
                if photo in records:
                    raise RuntimeError('duplicate_crop_ownership')
                if value['patient_key'] != accepted[photo]['patient_key']:
                    raise RuntimeError('crop_patient_identity_conflict')
                records[photo] = value
        finally:
            db.close()
    if set(records) != set(accepted):
        raise RuntimeError('final_accepted_crop_coverage_incomplete')
    fields = ['photo_id', 'patient_key', 'original_path', 'crop_path', 'view',
              'width', 'height', 'crop_sha256', 'status', 'reason', 'training_admitted']
    counts = {}
    for name, want_success in (('cropped_candidates.tsv', True), ('crop_hold.tsv', False)):
        target = run/name
        temporary = target.with_name(target.name+'.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for photo, value in sorted(records.items()):
                success = value['status'] == 'validated_provisional'
                if success != want_success:
                    continue
                row = accepted[photo]
                output = value.get('output', {})
                crop_path = root/value['output_path'] if success else None
                if success and (not crop_path.is_file() or crop_path.stat().st_size == 0):
                    raise RuntimeError('saved_crop_missing_or_changed')
                writer.writerow({'photo_id':photo, 'patient_key':row['patient_key'],
                    'original_path':row['original_path'], 'crop_path':str(crop_path) if success else '',
                    'view':value['view'], 'width':output.get('width', ''), 'height':output.get('height', ''),
                    'crop_sha256':output.get('sha256', ''), 'status':value['status'],
                    'reason':value.get('reason', ''), 'training_admitted':'false'})
                counts[value['status']] = counts.get(value['status'], 0)+1
        temporary.replace(target)
    save(run/'crop_completion.json', {'state':'crop_complete', 'completed_at':time.time(),
        'accepted_input_tasks':len(accepted), 'counts':counts,
        'provisional_records_excluded_by_final_annotation':extra,
        'final_annotation_reconciled':True, 'original_images_untouched':True,
        'training_admitted':False, 'independent_crop_review_complete':False,
        'next_stage':'independent_crop_quality_review_and_patient_source_split'})
    return True


def tick(root):
    run = root.parent
    marker = load(root/'READY.json')
    if not marker:
        save(root/'handoff_status.json', {'state':'waiting_for_crop_environment', 'updated_at':time.time()})
        return
    count = int(marker.get('worker_count', 2))
    ensure_crop_capacity(root, count)
    jobs, busy = scheduler()
    launches = load(root/'handoff_launches.json')
    states = [load(root/'workers'/f'slot_{i:02d}.json') for i in range(count)]
    completed = completed_shards(run)
    reserved = set()
    for key, entry in launches.items():
        job = entry['allocation_job_id']
        state = states[int(key)].get('state')
        if job in busy:
            reserved.add(job)
        elif entry.get('state') == 'launched' and time.time()-entry['launched_at'] < 90:
            reserved.add(job)
        if state in TERMINAL_HOLDS:
            entry['state'] = 'held_for_review'
        elif job not in reserved and entry.get('state') == 'launched':
            if state in ('finished_provisional', 'released_waiting_for_annotations_or_pilot'):
                entry['state'] = 'finished' if state == 'finished_provisional' else 'awaiting_more_annotations'
                entry['failed_launches'] = 0
            else:
                entry['state'] = 'launcher_exited_before_completion'
                entry['failed_launches'] = int(entry.get('failed_launches', 0))+1
    idle = idle_allocations(run, allocations(run), jobs, busy, reserved)
    for slot, state in enumerate(states):
        key = str(slot)
        prior = launches.get(key, {})
        worker_state = state.get('state')
        if worker_state == 'finished_provisional' or worker_state in TERMINAL_HOLDS:
            continue
        if prior.get('allocation_job_id') in reserved:
            continue
        if worker_state == 'released_waiting_for_annotations_or_pilot' and completed <= prior.get('completed_shards_at_launch', 0):
            continue
        attempts = int(prior.get('attempts', 0))
        if int(prior.get('failed_launches', 0)) >= 3:
            prior['state'] = 'held_launch_failure'
            launches[key] = prior
            continue
        if not idle:
            continue
        allocation = idle.pop(0)
        job = allocation['job']
        command = [SLURM+'srun', '--jobid='+job, '--nodes=1', '--ntasks=1',
                   '--cpus-per-task=4', '--gpus-per-task=1', '--gpu-bind=single:1',
                   '--exclusive', '--export=ALL', '/bin/bash', str(root/'launch_crop.sh'), str(slot)]
        save(root/'workers'/f'slot_{slot:02d}.json', {'state':'queued_in_existing_allocation',
             'slot':slot, 'allocation_job_id':job, 'updated_at':time.time()})
        with (root/'workers'/f'slot_{slot:02d}.launch.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        launches[key] = {'state':'launched', 'allocation_job_id':job, 'replica_slot':allocation['replica'],
                         'launcher_pid':process.pid, 'attempts':attempts+1, 'launched_at':time.time(),
                         'failed_launches':int(prior.get('failed_launches', 0)),
                         'completed_shards_at_launch':completed}
        reserved.add(job)
    save(root/'handoff_launches.json', launches)
    pilot = [load(root/'pilot'/f'slot_{i:02d}.json') for i in range(count)]
    if any(load(root/'gpu_smoke'/f'slot_{i:02d}.json').get('state') == 'synthetic_gpu_checks_passed'
           for i in range(count)):
        notify(root, 'crop GPU checks completed', 'Crop GPU startup checks',
               'The local GPU detector and crop verifier passed synthetic startup checks. A clinical crop pilot is next or in progress. Images and individual results remain on HPC; no training release has been declared.')
    if all(x.get('passed') is True for x in pilot):
        notify(root, 'crop pilot completed', 'Automated crop pilot',
               'The local automated crop pilot passed for frontal, profile and oblique views. Bulk processing is continuing. Crops remain provisional pending independent quality review.')
    held = sum(x.get('state') in TERMINAL_HOLDS for x in states)
    held_launch = sum(x.get('state') == 'held_launch_failure' for x in launches.values())
    if held or held_launch:
        notify(root, 'crop attention required', 'Crop processing requires attention',
               'A local cropping worker or launcher requires review. Other processing may continue. Private diagnostics remain on HPC; training data have not been released.')
    final = finalize(root, count)
    save(root/'handoff_status.json', {'state':'crop_complete' if final else 'running_with_hold' if held or held_launch
         else 'handoff_active' if launches else 'waiting_for_gpu_allocation'
         if load(run/'completion.json').get('state') == 'annotation_complete'
         else 'waiting_for_annotation_worker_to_finish',
         'worker_count':count, 'completed_annotation_shards':completed,
         'launched_crop_workers':len(launches), 'held_workers':held+held_launch,
         'keep_allocations':True, 'updated_at':time.time()})


def main(root, once):
    lock = (root/'handoff.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        return
    while True:
        try:
            tick(root)
        except Exception as exc:
            save(root/'handoff_status.json', {'state':'controller_needs_attention',
                 'error_type':type(exc).__name__, 'updated_at':time.time()})
            # Traceback and exception strings can contain private paths: retain locally only.
            import traceback
            with (root/'handoff.private_errors.log').open('a') as handle:
                traceback.print_exc(file=handle)
        if once or load(root.parent/'crop_completion.json').get('state') == 'crop_complete':
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(ROOT, args.once)
