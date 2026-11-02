"""Local adapters around the pinned FINO loop; validation blocks optimizer progress."""
import hashlib
import os
import random
import time
from pathlib import Path
from configuration import WORK, BASE, V1, DIAGNOSTICS, VENDOR, SEED, BACKBONE_LR_RATIO, STEPS
import sys
sys.path[:0] = [str(VENDOR), str(BASE), str(V1), str(DIAGNOSTICS), str(WORK)]
import numpy as np
import torch
import torch.distributed as dist
from common import load, save, now, install_rope_dtype_compatibility
from dinov3.train import train as official
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dataset import make_dataset as released_dataset
from warmup_rank import SyntheticDataset
from head_initialization import load_heads_only
from configuration import WARMED_CHECKPOINT


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
    rank, world = int(os.environ['COMPARISON_RANK']), int(os.environ['COMPARISON_WORLD'])
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                      MASTER_ADDR=os.environ['COMPARISON_MASTER'], MASTER_PORT=os.environ['COMPARISON_PORT'],
                      TORCHELASTIC_RUN_ID='cranio-controlled-'+os.environ['COMPARISON_PORT'])
    os.umask(0o077)
    torch.set_num_threads(4)
    install_rope_dtype_compatibility()
    folder = WORK/arm/phase
    original_init, original_groups = SSLMetaArch.init_weights, SSLMetaArch.get_maybe_fused_params_for_submodel
    original_checkpoint, original_forward = official.save_checkpoint, SSLMetaArch.forward_backward
    model_ref = {}
    state = {'state':'running', 'completed_steps':0}
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
        if arm == 'warmed_heads':
            load_heads_only(self, WARMED_CHECKPOINT)
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
        if phase == 'training':
            from validation import validate
            if not validate(self.cfg, self, 0, folder):
                raise RuntimeError('initial_validation_guard_failed')
        # Identical RNG state immediately before official dataloader construction.
        random.seed(SEED+rank); np.random.seed(SEED+rank)
        torch.manual_seed(SEED+rank); torch.cuda.manual_seed_all(SEED+rank)
        torch.cuda.reset_peak_memory_stats()

    def checked_forward(self, data, *args, **kwargs):
        step = kwargs['iteration']+1
        if phase == 'training':
            checksum = input_hash(data)
            reference_file = WORK/'warmed_heads/training/input_hashes'/f'rank_{rank:02d}_step_{step:06d}.json'
            valid = torch.ones((), device='cuda', dtype=torch.int)
            if arm == 'random_heads' and reference_file.exists():
                valid.fill_(int(load(reference_file).get('sha256') == checksum))
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            if not bool(valid):
                raise RuntimeError('paired_training_batch_or_augmentation_mismatch')
            save(folder/'input_hashes'/f'rank_{rank:02d}_step_{step:06d}.json', {'sha256':checksum})
        loss, metrics = original_forward(self, data, *args, **kwargs)
        finite = torch.isfinite(loss).all().to(torch.int)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError('nonfinite_loss_before_optimizer_step')
        return loss, metrics

    def checkpointed(ckpt_dir, *args, **kwargs):
        original_checkpoint(ckpt_dir, *args, **kwargs)
        step = int(Path(ckpt_dir).name)+1
        state['completed_steps'] = step
        if phase == 'capacity':
            return
        from validation import validate
        passed = validate(model_ref['model'].cfg, model_ref['model'], step, folder)
        if not passed:
            state['state'] = 'paused_validation_regression'
            raise BoundedStop()
        if step >= STEPS:
            state['state'] = 'bounded_pilot_complete_pending_review'
            raise BoundedStop()

    def registered_dataset(*, dataset_str, transform, target_transform=None, **kwargs):
        if phase == 'capacity':
            return SyntheticDataset(transform, target_transform)
        return released_dataset(dataset_str=dataset_str, transform=transform, target_transform=target_transform, **kwargs)

    official.make_dataset = registered_dataset
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
