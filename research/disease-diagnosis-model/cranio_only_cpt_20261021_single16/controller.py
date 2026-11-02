"""One clinical-only sixteen-GPU model with strict resume and validation guards."""
import concurrent.futures
import contextlib
import fcntl
import os
import signal
import subprocess
import threading
import time
from configuration import WORK, BASE, V1, VENDOR, DIAGNOSTICS, ARMS, WORLD, STEPS, SEED, SEEDS, configuration, protocol, RESUME_STEP, SOURCE_TRAINING
from common import SLURM, PYTHON, environment, load, save, now, notify, digest
from runtime import claim_jobs
from allocation_pool import idle_pools, SchedulerUnavailable
from omegaconf import OmegaConf


def verify_preflight():
    proof = load(WORK/'preflight.json')
    if proof.get('state') != 'passed' or proof.get('tests_passed', 0) < 16:
        raise RuntimeError('preflight_required')
    for name, expected in proof['tested_file_sha256'].items():
        if digest(WORK/name) != expected:
            raise RuntimeError('code_changed_after_preflight')
    for name, expected in proof.get('external_source_sha256', {}).items():
        if digest(name) != expected:
            raise RuntimeError('dependency_changed_after_preflight')


def verify_source_snapshot():
    plan=load(WORK/'resume_plan.json')
    if plan.get('state')!='source_preserved' or plan.get('completed_steps')!=RESUME_STEP:
        raise RuntimeError('source_snapshot_required')
    for relative,expected in plan['files'].items():
        f=SOURCE_TRAINING/relative
        if not f.is_file() or digest(f)!=expected:raise RuntimeError('source_snapshot_modified')
    for relative,expected in plan['checkpoint_file_sizes'].items():
        f=SOURCE_TRAINING/relative
        if not f.is_file() or f.stat().st_size!=expected:raise RuntimeError('checkpoint_shard_missing_or_truncated')
    report=load(SOURCE_TRAINING/'validation'/f'step_{RESUME_STEP:06d}'/'report.json')
    if report.get('state')!='complete' or not report['decision']['continue_training']:
        raise RuntimeError('source_checkpoint_failed_guard')


def audit_sources(arm):
    rows = [load(WORK/arm/'training'/f'dataset_source_rank_{rank:02d}.json') for rank in range(WORLD)]
    if not all(r.get('clinical_only') and r.get('reference_manifest_read') is False and r.get('split') == 'train' for r in rows):
        raise RuntimeError('clinical_source_receipt_missing')
    return dict(state='passed', clinical_only=True, ranks_verified=WORLD)


def stop_clients(clients):
    for client in clients:
        if client.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(client.pid, signal.SIGTERM)
    for client in clients:
        try:
            client.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(client.pid, signal.SIGKILL)
            client.wait(timeout=15)


def run_probe(pool, port):
    clients=[]
    folder=WORK/'gpu_probe'
    folder.mkdir(exist_ok=True)
    try:
        for rank,allocation in enumerate(pool):
            env=environment()
            env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,VENDOR,BASE,V1,DIAGNOSTICS]))
            env.update(RANK=str(rank),WORLD_SIZE=str(WORLD),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',
                MASTER_ADDR=pool[0]['node'],MASTER_PORT=str(port+1),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            cmd=[str(SLURM/'srun'),'--jobid='+allocation['job_id'],'--nodes=1','--ntasks=1',
                '--cpus-per-task=1','--mem=4G','--gpus-per-task=1','--gpu-bind=single:1',
                '--cpu-bind=none','--exact','--overlap' if allocation['step_overlap'] else '--exclusive',
                '--mpi=none','--time=00:15:00',str(PYTHON),str(WORK/'gpu_probe.py')]
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True))
        deadline=time.monotonic()+12*60
        while True:
            codes=[p.poll() for p in clients]
            if any(c not in (None,0) for c in codes):raise RuntimeError('gpu_communication_or_reshard_probe_failed')
            if all(c==0 for c in codes):break
            if time.monotonic()>deadline:raise RuntimeError('gpu_probe_timeout')
            time.sleep(2)
        rows=[load(folder/f'rank_{i:02d}.json') for i in range(WORLD)]
        if not all(r.get('state')=='passed' and r.get('world_size')==WORLD and r.get('optimizer_reshard_next_update_matches') for r in rows):
            raise RuntimeError('gpu_probe_receipts_incomplete')
        save(folder/'completion.json',dict(state='passed',world_size=WORLD,
            maximum_64MiB_allreduce_seconds=max(r['allreduce_seconds'] for r in rows),
            full_state_reshard='8 to 16 ranks; synthetic model, EMA teacher and AdamW next update verified',
            completed_at_utc=now()))
    finally:
        stop_clients(clients)


def run_arm(arm, phase, pool, abort):
    folder = WORK/arm/phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'ckpt').exists() or (folder/'rank_00_receipt.json').exists():
        raise RuntimeError('existing_experiment_requires_review')
    OmegaConf.save(configuration(arm, phase), folder/'config.yaml')
    clients = []
    try:
        port = 30000+(int(os.environ['SLURM_JOB_ID']) % 10000)*2 + ARMS.index(arm)
        save(WORK/'status.json',dict(state='running',phase='synthetic_gpu_reshard_probe',
             active_gpus=WORLD,independent_models=1,updated_at_utc=now()))
        run_probe(pool, port)
        save(WORK/'status.json',dict(state='running',phase='full_checkpoint_restore_and_training',
             active_gpus=WORLD,independent_models=1,updated_at_utc=now()))
        for rank, allocation in enumerate(pool):
            env = environment()
            env['PYTHONPATH'] = os.pathsep.join(map(str, [WORK, VENDOR, BASE, V1, DIAGNOSTICS]))
            env.update(COMPARISON_ARM=arm, COMPARISON_PHASE=phase, COMPARISON_RANK=str(rank),
                       COMPARISON_WORLD=str(WORLD), COMPARISON_MASTER=pool[0]['node'], COMPARISON_PORT=str(port), COMPARISON_SEED=str(SEEDS[arm]),
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            command = [str(SLURM/'srun'), '--jobid='+allocation['job_id'], '--nodes=1', '--ntasks=1',
                       '--job-name=cranio-single16-rank',
                       '--cpus-per-task=4', f"--mem={allocation['host_memory_gib']}G", '--gpus-per-task=1',
                       '--gpu-bind=single:1', '--cpu-bind=none', '--exact',
                       '--overlap' if allocation['step_overlap'] else '--exclusive', '--mpi=none', '--time=10:00:00',
                       str(PYTHON), str(WORK/'rank.py'), '--config-file', str(folder/'config.yaml'),
                       '--output-dir', str(folder), '--seed', str(SEEDS[arm])]
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as stream:
                clients.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=stream, start_new_session=True))
        save(folder/'execution.json', dict(state='running', arm=arm, phase=phase, world_size=WORLD, updated_at_utc=now()))
        deadline = time.monotonic()+9.8*3600
        progress, previous_marker = time.monotonic(), None
        while True:
            codes = [p.poll() for p in clients]
            if abort.is_set():
                raise RuntimeError('peer_execution_failure')
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('worker_failure')
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('bounded_execution_deadline')
            paths = list(folder.glob('initialization_rank_*.json')) + list(folder.glob('ckpt/*/.metadata'))
            paths += list(folder.glob('validation/step_*/report.json'))
            if (folder/'progress.json').exists(): paths.append(folder/'progress.json')
            marker = max((p.stat().st_mtime for p in paths), default=None)
            if marker != previous_marker:
                previous_marker, progress = marker, time.monotonic()
            if time.monotonic()-progress > 30*60:
                raise RuntimeError('no_verified_progress_for_30_minutes')
            time.sleep(5)
        receipts = [load(folder/f'rank_{rank:02d}_receipt.json') for rank in range(WORLD)]
        if not all(r.get('success') for r in receipts) or len({(r.get('state'),r.get('completed_steps')) for r in receipts}) != 1:
            raise RuntimeError('incomplete_or_inconsistent_rank_receipts')
        if phase == 'capacity' and not all(r['peak_reserved_GiB'] < 72 and r['free_GiB'] >= 6 for r in receipts):
            raise RuntimeError('capacity_gpu_headroom_failed')
        telemetry = [load(folder/'telemetry'/f'rank_{rank:02d}.json') for rank in range(WORLD)]
        if phase == 'capacity' and not all(t.get('cgroups') and t['max_host_resident_fraction'] < .85 for t in telemetry):
            raise RuntimeError('capacity_host_headroom_failed')
        receipt = dict(state=receipts[0]['state'], completed_steps=receipts[0]['completed_steps'],
                       world_size=WORLD, max_reserved_GiB=max(r['peak_reserved_GiB'] for r in receipts),
                       completed_at_utc=now(), test_used=False)
        save(folder/'completion.json', receipt)
        save(folder/'execution.json', dict(receipt, arm=arm, phase=phase))
        if phase == 'training':
            notify('single16 '+arm+' '+receipt['state'], 'Craniofacial single-model training finished or paused',
                   'The single sixteen-GPU model completed or paused at a validation guard. Checkpoints and aggregate reports remain on HPC.')
        return receipt
    except BaseException as error:
        abort.set()
        save(folder/'execution.json', dict(state='failed', error_type=type(error).__name__, updated_at_utc=now()))
        raise
    finally:
        stop_clients(clients)


def main():
    os.umask(0o077)
    verify_preflight()
    verify_source_snapshot()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline = time.monotonic()+30*60
        while True:
            try:
                pools = idle_pools()
                if all(len(pools[a]) == WORLD for a in ARMS): break
            except SchedulerUnavailable:
                pass
            if time.monotonic() > deadline: raise RuntimeError('allocation_readiness_timeout')
            save(WORK/'status.json', dict(state='waiting_for_selected_allocations', updated_at_utc=now()))
            time.sleep(30)
        locks = claim_jobs([p['job_id'] for arm in ARMS for p in pools[arm]])
        try:
            verified = idle_pools()
            if verified != pools: raise RuntimeError('allocation_changed_before_launch')
            save(WORK/'active_pools.json', dict(arms=pools, selected_at_utc=now()))
            results = {}
            for phase in ('training',):
                save(WORK/'status.json', dict(state='running', phase=phase, active_gpus=16,
                     world_size_per_arm=WORLD, independent_models=1, automatic_scale_up=False, updated_at_utc=now()))
                abort = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    futures = {arm:executor.submit(run_arm, arm, phase, pools[arm], abort) for arm in ARMS}
                    results[phase] = {arm:futures[arm].result() for arm in ARMS}
            audit = {arm:audit_sources(arm) for arm in ARMS}
            save(WORK/'status.json', dict(state='single_model_complete_or_guard_paused', results=results,
                 clinical_source_audit=audit, automatic_scale_up=False, completed_at_utc=now()))
            notify('single16 complete', 'Craniofacial sixteen-GPU training reached its review checkpoint',
                   'The single clinical-only model completed its bounded training budget or paused at a validation guard. Checkpoints and reports remain on HPC.')
        finally:
            for held in locks: held.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK/'status.json', dict(state='paused_execution_or_integrity_failure',
             error_type=type(error).__name__, automatic_restart=False, updated_at_utc=now()))
        notify('clinical-only single16 execution failure', 'Craniofacial continuation requires inspection',
               'The single-model run stopped on a local execution, integrity or memory check. Completed checkpoints are preserved. No automatic retry or expansion is enabled.')
        raise

