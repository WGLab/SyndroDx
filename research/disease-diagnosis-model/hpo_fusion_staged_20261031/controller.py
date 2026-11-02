"""Dependency-aware public-pretrain -> GMDB adaptation, using free allocations."""
import fcntl
import subprocess
import time
from common import *

PILOT = PROJECT / 'hpo_fusion_public_20261031'


def launch(job, script, args, name):
    cmd = [str(SLURM / 'srun'), '--jobid=' + job['id'], '--job-name=' + name,
           '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=48G', '--gpus-per-task=1',
           '--overlap', '--exact', '--immediate=30', '--mpi=none', str(PYTHON), str(WORK / script), *args]
    with (WORK / 'logs' / (name + '.private.log')).open('ab') as f:
        return subprocess.Popen(cmd, env=environment(), stdout=f, stderr=f)


def main():
    os.umask(0o077)
    lock = (WORK / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, sha in load(WORK / 'source_receipt.json').items():
        assert digest(WORK / name) == sha
    protocol = load(WORK / 'protocol.json')
    cases = load(WORK / 'cases.json')['cases']
    pretrains = [condition + '__seed' + str(seed) for seed in protocol['seeds']
                for condition in protocol['conditions'] if condition != 'gmdb_only']
    active, retired, attempts = {}, {}, {}
    while True:
        # Exact controls already running in the first study are reused, not trained twice.
        for n, case in cases.items():
            if case['condition'] != 'gmdb_only' or (WORK / 'cases' / n / 'result.json').exists():
                continue
            report = load(PILOT / 'cases' / n / 'result.json')
            if report.get('state') == 'complete':
                folder = WORK / 'cases' / n
                folder.mkdir(exist_ok=True)
                report['reused_from'] = str(PILOT / 'cases' / n / 'result.json')
                save(folder / 'result.json', report)
                target = folder / 'validation_logits.private.npy'
                if not target.exists():
                    target.symlink_to(PILOT / 'cases' / n / 'validation_logits.private.npy')
        done = {n for n in cases if load(WORK / 'cases' / n / 'result.json').get('state') == 'complete'}
        pre_done = {n for n in pretrains if load(WORK / 'pretrain' / n / 'result.json').get('state') == 'complete'}
        for key, process in list(active.items()):
            code = process.poll()
            if code is not None:
                del active[key]
                retired[key] = time.monotonic()
                if code and attempts[key] >= 3:
                    raise RuntimeError('repeated_staged_worker_failure')
        if len(done) == len(cases):
            break
        try:
            jobs = free_jobs()
        except Exception:
            jobs = []
        tasks = [('pre:' + n, 'pretrain_hpo.py', n) for n in pretrains if n not in pre_done]
        tasks += [('case:' + n, 'train.py', n) for n, c in cases.items()
                  if n not in done and c['condition'] != 'gmdb_only'
                  and c['condition'] + '__seed' + str(c['seed']) in pre_done]
        for key, script, name in tasks:
            if not jobs:
                break
            if key in active or time.monotonic() - retired.get(key, -1e9) < 180:
                continue
            attempts[key] = attempts.get(key, 0) + 1
            active[key] = launch(jobs.pop(0), script, [name], 'hpo-staged-' + name)
        save(WORK / 'status.json', dict(state='public_pretrain_then_GMDB_adaptation',
             public_pretraining_complete=len(pre_done), public_pretraining_total=len(pretrains),
             diagnosis_complete=len(done), diagnosis_total=len(cases), active=list(active), updated_at_utc=now()))
        time.sleep(15)
    with (WORK / 'logs/statistics.private.log').open('ab') as f:
        subprocess.run([str(PYTHON), str(WORK / 'summarize.py')], env=environment(), stdout=f, stderr=f, check=True)
    save(WORK / 'status.json', dict(state='complete', updated_at_utc=now()))
    notify('staged public-data comparison complete', __import__('json').dumps(load(WORK / 'results.json'), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(WORK / 'controller_failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))
        notify('staged experiment requires inspection', type(e).__name__ + '; checkpoints and allocations retained.')
        raise
