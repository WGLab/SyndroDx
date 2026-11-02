"""Local adapters around the pinned FINO loop; validation blocks optimizer progress."""
import hashlib
import math
import gc
import os
import random
import time
from pathlib import Path
from configuration import WORK, BASE, V1, DIAGNOSTICS, VENDOR, SEED, BACKBONE_LR_RATIO, STEPS, RESUME_STEP, SOURCE_TRAINING, SOURCE_WORLD
import sys
sys.path[:0] = [str(WORK), str(VENDOR), str(BASE), str(V1), str(DIAGNOSTICS)]
import numpy as np
import torch
import torch.distributed as dist
from common import load, save, now, install_rope_dtype_compatibility, notify
from dinov3.train import train as official
from dinov3.train.ssl_meta_arch import SSLMetaArch
from clinical_dataset import make_dataset as released_dataset
from warmup_rank import SyntheticDataset
from head_initialization import load_heads_only
from configuration import WARMED_CHECKPOINT
from telemetry import Monitor, memory_snapshot
from continuation import loader_options,restore_full_state,compare_resharded_features


class BoundedStop(Exception):
    pass


def fingerprint(module):
    result = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        result.update(name.encode())
        result.update(value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def input_hash(data):
    result = hashlib.sha256()
    for key in sorted(data):
        value = data[key]
        if isinstance(value, torch.Tensor):
            result.update(key.encode())
            result.update(str(tuple(value.shape)).encode())
            result.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    arm, phase = os.environ['COMPARISON_ARM'], os.environ['COMPARISON_PHASE']
    seed = int(os.environ['COMPARISON_SEED'])
    rank, world = int(os.environ['COMPARISON_RANK']), int(os.environ['COMPARISON_WORLD'])
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                      MASTER_ADDR=os.environ['COMPARISON_MASTER'], MASTER_PORT=os.environ['COMPARISON_PORT'],
                      TORCHELASTIC_RUN_ID='cranio-controlled-'+os.environ['COMPARISON_PORT'])
    os.umask(0o077)
    torch.set_num_threads(4)
    install_rope_dtype_compatibility()
    folder = WORK/arm/phase
    monitor = Monitor(folder/'telemetry', rank)
    monitor.start()
    original_init, original_groups = SSLMetaArch.init_weights, SSLMetaArch.get_maybe_fused_params_for_submodel
    original_checkpoint, original_forward = official.save_checkpoint, SSLMetaArch.forward_backward
    original_loader,original_restore=official.make_data_loader,official.load_checkpoint
    source=SOURCE_TRAINING
    model_ref = {}
    state = {'state':'running', 'completed_steps':RESUME_STEP}
    reference = torch.load(BASE/'initialization/public_synthetic_reference.pt', map_location='cpu', weights_only=True)

    def grouped(self, module):
        groups = list(original_groups(self, module))
        if hasattr(getattr(module, 'module', module), 'blocks'):
            if not all(group['is_frozen_backbone'] for group in groups):
                raise RuntimeError('backbone_group_annotation_missing')
            for group in groups:
                group['lr_multiplier'] *= BACKBONE_LR_RATIO
        return groups

    def initialized(self):
        original_init(self)
        before = {owner: fingerprint(getattr(self, owner).backbone) for owner in ('student', 'teacher')}
        after = {owner: fingerprint(getattr(self, owner).backbone) for owner in ('student', 'teacher')}
        if before != after:
            raise RuntimeError('head_loading_modified_backbone')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            actual = self.teacher.backbone(reference['input'].cuda(), is_training=True)['x_norm_clstoken'].float()
        cosine = torch.nn.functional.cosine_similarity(actual, reference['expected_cls'].cuda()).min()
        dist.all_reduce(cosine, op=dist.ReduceOp.MIN)
        if not bool(torch.isfinite(cosine)) or float(cosine) < .999:
            raise RuntimeError('original_backbone_initialization_mismatch')
        save(folder/f'initialization_rank_{rank:02d}.json', dict(state='passed',
             original_backbones_preserved=True, synthetic_reference_cosine=float(cosine)))
        model_ref['model'] = self
        gc.collect()
        dist.barrier()
        # Identical RNG state immediately before official dataloader construction.
        random.seed(seed+rank); np.random.seed(seed+rank)
        torch.manual_seed(seed+rank); torch.cuda.manual_seed_all(seed+rank)
        torch.cuda.reset_peak_memory_stats()

    def continued_loader(*args,**kwargs):
        options=loader_options(kwargs,RESUME_STEP)
        return original_loader(*args,**options)

    def restored(path,*,model,optimizer=None,**kwargs):
        iteration,proof=restore_full_state(path,model=model,optimizer=optimizer,loader=original_restore,
                                          completed_steps=RESUME_STEP,**kwargs)
        from validation import validate
        if not validate(model.cfg,model,RESUME_STEP,folder):
            raise RuntimeError('restored_checkpoint_validation_guard_failed')
        report=load(folder/'validation'/f'step_{RESUME_STEP:06d}'/'report.json')
        reference_report=load(source/'validation'/f'step_{RESUME_STEP:06d}'/'report.json')
        if any(abs(report['metrics'][k]-v)>1e-12 for k,v in reference_report['metrics'].items()):
            raise RuntimeError('restored_checkpoint_metrics_mismatch')
        valid=torch.ones((),device='cuda',dtype=torch.int)
        minimum=1.
        if rank==0:
            try:
                minimum=compare_resharded_features(source/'validation'/f'step_{RESUME_STEP:06d}',
                    folder/'validation'/f'step_{RESUME_STEP:06d}',SOURCE_WORLD,world)
                valid.fill_(int(math.isfinite(minimum) and minimum>.99999))
            except Exception:
                valid.zero_()
        dist.broadcast(valid,src=0)
        if not bool(valid):raise RuntimeError('restored_checkpoint_features_mismatch')
        save(folder/f'resume_rank_{rank:02d}.json',dict(state='passed',resumed_completed_steps=RESUME_STEP,
             teacher_features_match_source=True,source_world=SOURCE_WORLD,target_world=world,**proof))
        # Worker augmentation RNG was not in the old checkpoint. Use a declared,
        # reproducible continuation seed rather than claim bitwise replay.
        random.seed(seed+RESUME_STEP+rank);np.random.seed(seed+RESUME_STEP+rank)
        torch.manual_seed(seed+RESUME_STEP+rank);torch.cuda.manual_seed_all(seed+RESUME_STEP+rank)
        gc.collect();dist.barrier()
        return iteration

    def checked_forward(self, data, *args, **kwargs):
        step = kwargs['iteration']+1
        if (step-1) % 10 == 0:
            gc.collect()
        safe = torch.tensor(int(memory_snapshot()['max_host_resident_fraction'] < .92), device='cuda')
        dist.all_reduce(safe, op=dist.ReduceOp.MIN)
        if not bool(safe):
            raise RuntimeError('host_memory_headroom_guard')
        if phase == 'training':
            checksum = input_hash(data)
            save(folder/'input_hashes'/f'rank_{rank:02d}_step_{step:06d}.json', {'sha256':checksum})
        if rank == 0:
            save(folder/'progress.json', dict(state='optimizer_iteration_started', iteration=step-1,
                                             updated_at_utc=now()))
        loss, metrics = original_forward(self, data, *args, **kwargs)
        finite = torch.isfinite(loss).all().to(torch.int)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError('nonfinite_loss_before_optimizer_step')
        return loss, metrics

    def checkpointed(ckpt_dir, *args, **kwargs):
        gc.collect()
        dist.barrier()
        original_checkpoint(ckpt_dir, *args, **kwargs)
        gc.collect()
        dist.barrier()
        step = int(Path(ckpt_dir).name)+1
        state['completed_steps'] = step
        if phase == 'capacity':
            return
        from validation import validate
        passed = validate(model_ref['model'].cfg, model_ref['model'], step, folder)
        gc.collect()
        dist.barrier()
        if not passed:
            state['state'] = load(folder/'validation'/f'step_{step:06d}'/'report.json')['decision']['state']
            raise BoundedStop()
        requested=torch.tensor(int((WORK/'stop_requested.json').exists()),device='cuda')
        dist.all_reduce(requested,op=dist.ReduceOp.MAX)
        if bool(requested):
            state['state']='paused_walltime_checkpoint_preserved'
            raise BoundedStop()
        if step==1000 and rank==0:
            notify('resume20 milestone1000','Craniofacial pretraining reached its intermediate checkpoint',
                   'The single clinical-only model reached its intermediate checkpoint. Frozen-probe results and original-model comparisons remain on HPC. Training continues only while the declared guards pass.')
        if step >= STEPS:
            state['state'] = 'bounded_training_complete_pending_review'
            raise BoundedStop()

    def registered_dataset(*, dataset_str, transform, target_transform=None, **kwargs):
        if phase == 'capacity':
            return SyntheticDataset(transform, target_transform)
        dataset = released_dataset(dataset_str=dataset_str, transform=transform, target_transform=target_transform, **kwargs)
        save(folder/f'dataset_source_rank_{rank:02d}.json',dict(clinical_only=True,reference_manifest_read=False,
             training_images=dataset.image_count,training_patient_groups=len(dataset.groups),split='train'))
        return dataset

    official.make_dataset = registered_dataset
    official.make_data_loader = continued_loader
    official.find_latest_checkpoint = lambda directory:source/'ckpt'/str(RESUME_STEP-1)
    official.load_checkpoint = restored
    SSLMetaArch.init_weights = initialized
    SSLMetaArch.get_maybe_fused_params_for_submodel = grouped
    SSLMetaArch.forward_backward = checked_forward
    official.save_checkpoint = checkpointed
    started, success, error = time.monotonic(), False, None
    try:
        official.main()
        state['state'] = 'capacity_passed' if phase == 'capacity' else 'bounded_pilot_complete_pending_review'
        success = True
    except BoundedStop:
        success = True
    except Exception as failure:
        state['state'], error = 'failed', failure
        raise
    finally:
        monitor.close()
        receipt = dict(state, success=success, rank=rank, arm=arm, phase=phase,
                       error_type=type(error).__name__ if error else None,
                       clinical_data_accessed=phase == 'training', elapsed_seconds=time.monotonic()-started,
                       completed_at_utc=now())
        if torch.cuda.is_initialized():
            free, _ = torch.cuda.mem_get_info()
            receipt.update(peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                           free_GiB=free/2**30)
        save(folder/f'rank_{rank:02d}_receipt.json', receipt)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
