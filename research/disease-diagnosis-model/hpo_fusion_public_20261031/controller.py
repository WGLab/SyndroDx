"""Use available allocated GPUs, retain allocations, resume at epoch boundaries."""
import fcntl
import subprocess
import time
from common import *


def launch(job, script, args, name):
    cmd = [str(SLURM / 'srun'), '--jobid=' + job['id'], '--job-name=' + name,
           '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=48G',
           '--gpus-per-task=1', '--overlap', '--exact', '--immediate=30', '--mpi=none',
           str(PYTHON), str(WORK / script), *args]
    log = (WORK / 'logs' / (name + '.private.log')).open('ab')
    process = subprocess.Popen(cmd, env=environment(), stdout=log, stderr=log)
    log.close()
    return process


def main():
    os.umask(0o077)
    lock = (WORK / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, sha in load(WORK / 'source_receipt.json').items():
        assert digest(WORK / name) == sha
    assert load(WORK / 'data_receipt.json')['state'] == 'complete'
    assert load(WORK / 'tests.json')['state'] == 'passed'
    notify('data ready', __import__('json').dumps(load(WORK / 'data_receipt.json')['counts']))
    attempts = 0
    while load(WORK / 'term_embedding_receipt.json').get('state') != 'complete':
        try:
            jobs = free_jobs()
        except Exception:
            jobs = []
        if not jobs:
            save(WORK / 'status.json', dict(state='waiting_for_free_allocated_GPU_for_public_term_encoding', updated_at_utc=now()))
            time.sleep(15)
            continue
        process = launch(jobs[0], 'embed_terms.py', [], 'hpo-public-ontology-encoding')
        save(WORK / 'status.json', dict(state='encoding_public_ontology', gpu_job=jobs[0]['id'], updated_at_utc=now()))
        while process.poll() is None:
            time.sleep(10)
        # Shared metadata can lag worker completion.
        deadline = time.monotonic() + 180
        while not (WORK / 'term_embedding_receipt.json').exists() and time.monotonic() < deadline:
            time.sleep(5)
        if not (WORK / 'term_embedding_receipt.json').exists():
            attempts += 1
            if attempts >= 3:
                raise RuntimeError('ontology_encoding_failed_repeatedly')
    cases = load(WORK / 'cases.json')['cases']
    active, retired, attempts = {}, {}, {}
    while True:
        done = {n for n in cases if load(WORK / 'cases' / n / 'result.json').get('state') == 'complete'}
        for n, process in list(active.items()):
            code = process.poll()
            if code is not None:
                del active[n]
                retired[n] = time.monotonic()
                if code and attempts[n] >= 3 and n not in done:
                    raise RuntimeError('repeated_worker_failure')
        if len(done) == len(cases):
            break
        try:
            jobs = free_jobs()
        except Exception:
            jobs = []
        for n in cases:
            if not jobs:
                break
            if n in done or n in active or time.monotonic() - retired.get(n, -1e9) < 180:
                continue
            attempts[n] = attempts.get(n, 0) + 1
            active[n] = launch(jobs.pop(0), 'train.py', [n], 'hpo-fusion-' + n)
        save(WORK / 'status.json', dict(state='matched_diagnosis_training', complete=len(done), total=len(cases),
             active=list(active), updated_at_utc=now()))
        time.sleep(15)
    save(WORK / 'status.json', dict(state='paired_development_statistics', updated_at_utc=now()))
    with (WORK / 'logs/statistics.private.log').open('ab') as f:
        subprocess.run([str(PYTHON), str(WORK / 'summarize.py')], env=environment(), stdout=f, stderr=f, check=True)
    save(WORK / 'status.json', dict(state='complete', updated_at_utc=now()))
    notify('matched diagnosis complete', __import__('json').dumps(load(WORK / 'results.json'), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(WORK / 'controller_failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))
        notify('requires inspection', type(e).__name__ + '; checkpoints and allocations preserved.')
        raise
