"""Capacity-gated 200-step head-only pilot on idle existing H100 allocations."""
import contextlib
import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
WORK = Path(__file__).resolve().parent
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
sys.path.insert(0, str(BASE))
from common import V1, SLURM, PYTHON, environment, load, save, now, notify
from allocations import active_allocations
sys.path.insert(0, str(V1))
from runtime import claim_jobs
from warmup_config import configuration
from omegaconf import OmegaConf


def run(phase, pool, resume=False, stop_after=200):
    folder = WORK / 'warmup_pilot' / phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'ckpt').exists() and not resume:
        raise RuntimeError('existing_pilot_checkpoint_requires_explicit_resume_review')
    if resume:
        cfg = OmegaConf.load(folder/'config.yaml')
        if not (cfg.optim.epochs == 1000 and cfg.schedules.lr.freeze_student_iterations >= 1000
                and cfg.train.batch_size_per_gpu == 64 and len(pool) == 8
                and cfg.schedules.lr.freeze_patch_embed and cfg.schedules.lr.freeze_cls_token
                and cfg.schedules.lr.freeze_mask_token and stop_after == 1000):
            raise RuntimeError('frozen_resume_configuration_not_preserved')
    else:
        OmegaConf.save(configuration(len(pool), phase), folder/'config.yaml')
    clients = []
    try:
        for rank, allocation in enumerate(pool):
            env = environment()
            env['PYTHONPATH'] = os.pathsep.join(map(str, [WORK/'vendor/dinov3_fino', BASE, V1, WORK]))
            env.update(WARMUP_PHASE=phase, WARMUP_RANK=str(rank), WARMUP_WORLD=str(len(pool)),
                       WARMUP_STOP_AFTER=str(stop_after),
                       WARMUP_MASTER=pool[0]['node'], WARMUP_PORT=str(35000+int(os.environ['SLURM_JOB_ID'])%20000))
            command = [str(SLURM/'srun'), '--jobid='+allocation['job_id'], '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=4', '--mem-per-gpu=68G', '--gpus-per-task=1', '--gpu-bind=single:1',
                       '--cpu-bind=none', '--exact', '--exclusive', '--mpi=none', '--time=04:00:00' if resume else '--time=02:00:00',
                       str(PYTHON), str(WORK/'warmup_rank.py'), '--config-file', str(folder/'config.yaml'),
                       '--output-dir', str(folder), '--seed', '20261017']
            if not resume:
                command.append('--no-resume')
            with (folder/f'rank_{rank:02d}.private.log').open('ab') as output:
                clients.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                  stdout=output, stderr=output, start_new_session=True))
        save(WORK/'warmup_pilot/status.json', {'state':phase+'_running', 'world_size':len(pool),
             'backbone_unfreezing_permitted':False, 'maximum_training_steps':stop_after,
             'official_checkpoint_resume':resume, 'updated_at_utc':now()})
        deadline = time.monotonic()+(225 if resume else 110)*60
        while True:
            codes = [client.poll() for client in clients]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(phase+'_worker_failed')
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(phase+'_time_limit')
            time.sleep(5)
        receipts = [load(folder/f'rank_{rank}_receipt.json') for rank in range(len(pool))]
        if not all(receipt.get('state') == 'passed' for receipt in receipts):
            raise RuntimeError('incomplete_worker_receipts')
        if phase == 'capacity' and not all(receipt['peak_reserved_GiB'] < 72 and receipt['CUDA_free_GiB'] >= 6 for receipt in receipts):
            raise RuntimeError('synthetic_memory_headroom_guard_failed')
        if not load(folder/'checkpoint_status.json').get('all_student_backbone_shards_bitwise_unchanged'):
            raise RuntimeError('frozen_backbone_verification_missing')
    finally:
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


def main(resume_frozen=False):
    os.umask(0o077)
    pool, _ = active_allocations()
    result = subprocess.run([str(SLURM/'squeue'), '-h', '--steps', '-u', 'research_user', '-o', '%i'],
                  env=environment(), capture_output=True, text=True, check=True, timeout=30)
    busy = {line.split('.')[0] for line in result.stdout.splitlines()
            if '.' in line and not line.endswith(('.batch', '.extern'))}
    pool = [allocation for allocation in pool if allocation['job_id'] not in busy]
    if len(pool) < 8:
        raise RuntimeError('at_least_eight_idle_existing_H100_allocations_required')
    pool = pool[:12]
    locks = claim_jobs([allocation['job_id'] for allocation in pool])
    try:
        if resume_frozen:
            certificate = load(WORK/'warmup_pilot/capacity_passed.json')
            previous = load(WORK/'warmup_pilot/status.json')
            if certificate.get('state') != 'passed' or certificate.get('world_size') != len(pool) or previous.get('state') != 'pilot_complete' or previous.get('completed_steps') != 200:
                raise RuntimeError('completed_capacity_and_pilot_required_for_resume')
            save(WORK/'warmup_pilot/pilot200_completion.json', previous)
            preserved = WORK/'warmup_pilot/preserved_checkpoints/199'
            if not preserved.exists():
                # Completed DCP files are immutable; hard links retain the pilot
                # when the official rolling-checkpoint cleanup removes its path.
                shutil.copytree(WORK/'warmup_pilot/training/ckpt/199', preserved, copy_function=os.link)
            run('training', pool, resume=True, stop_after=1000)
        else:
            run('capacity', pool)
            save(WORK/'warmup_pilot/capacity_passed.json', {'state':'passed', 'world_size':len(pool),
                 'backbone_frozen_verified':True, 'synthetic_images_only':True, 'completed_at_utc':now()})
            run('training', pool)
        save(WORK/'warmup_pilot/status.json', {'state':'frozen_warmup1000_complete' if resume_frozen else 'pilot_complete', 'completed_steps':1000 if resume_frozen else 200,
             'world_size':len(pool), 'backbone_unfreezing_permitted':False,
             'backbone_frozen_verified':True, 'parents_retained':True, 'completed_at_utc':now()})
        notify('frozen-head warmup1000 complete' if resume_frozen else 'frozen-head pilot complete', 'Frozen-backbone head warmup completed',
               'The bounded official-FINO head-warmup pilot completed. The pretrained student backbone was verified unchanged at checkpoints. This is a head-initialization diagnostic, not evidence of improved clinical performance. Review the private head metrics before any unfreezing.')
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-frozen-to1000', action='store_true')
    args = parser.parse_args()
    try:
        main(args.resume_frozen_to1000)
    except Exception as error:
        save(WORK/'warmup_pilot/status.json', {'state':'failed', 'error_type':type(error).__name__,
             'parents_retained':True, 'updated_at_utc':now()})
        notify('frozen-head pilot failed', 'Frozen-backbone head warmup pilot requires review',
               'The bounded head-warmup pilot did not pass a local execution or acceptance check. Parent GPU allocations and completed encoders were retained. All detailed diagnostic information remains on HPC.')
        raise
