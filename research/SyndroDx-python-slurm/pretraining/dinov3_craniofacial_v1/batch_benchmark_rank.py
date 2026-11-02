"""Offline synthetic throughput benchmark matching the continual training loop."""
import argparse
import datetime
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from omegaconf import OmegaConf
import dinov3.distributed as distributed
import dinov3.train.train as upstream
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.data import collate_data_and_cast, MaskingGenerator
from runtime import save, now
from train_rank import ContinualMetaArch


class SyntheticArchitecture(ContinualMetaArch):
    def init_weights(self):
        # Same model/precision/optimizer, random weights, no clinical checkpoint.
        SSLMetaArch.init_weights(self)


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(20261015)
    distributed.enable(timeout=datetime.timedelta(seconds=600), restrict_print_to_main_process=False)
    rank = dist.get_rank()
    cfg = OmegaConf.load(args.config)
    cfg.train.batch_size_per_gpu = args.batch
    with torch.device('meta'):
        model = SyntheticArchitecture(cfg)
    model.prepare_for_distributed_training()
    model._apply(lambda t: torch.full_like(t, math.nan if t.dtype.is_floating_point else 0, device='cuda'), recurse=True)
    model.init_weights()
    model.train()
    optimizer = upstream.build_optimizer(cfg, model.get_params_groups())
    size = cfg.crops.global_crops_size
    local_size = cfg.crops.local_crops_size
    tokens = (size // cfg.student.patch_size) ** 2
    generator = MaskingGenerator((size // cfg.student.patch_size,) * 2, max_num_patches=tokens // 2)
    samples = [({'global_crops': [torch.randn(3, size, size) for _ in range(2)],
                 'local_crops': [torch.randn(3, local_size, local_size) for _ in range(cfg.crops.local_crops_number)]}, ())
               for _ in range(args.batch)]
    data = collate_data_and_cast(samples, tuple(cfg.ibot.mask_ratio_min_max), cfg.ibot.mask_sample_probability,
                                {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[cfg.compute_precision.param_dtype],
                                tokens, generator, random_circular_shift=cfg.ibot.mask_random_circular_shift)
    data['global_batch_size'] = args.batch * dist.get_world_size()
    times = []
    for iteration in range(8):
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        tick = time.monotonic()
        upstream.apply_optim_scheduler(optimizer, 1e-5, .04, 1e-5)
        loss, _ = model.forward_backward(data, teacher_temp=.07, iteration=iteration)
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError('nonfinite_synthetic_loss')
        for part in model.student.values():
            norm = torch.nn.utils.clip_grad_norm_(part.parameters(), cfg.optim.clip_grad)
            norm = norm.full_tensor() if isinstance(norm, DTensor) else norm
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('nonfinite_synthetic_gradient')
        optimizer.step()
        model.update_ema(.994)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.monotonic() - tick, device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if iteration >= 2:
            times.append(float(elapsed))
    save(Path(args.output) / f'rank_{rank:02d}.json', dict(
        state='passed', rank=rank, world_size=dist.get_world_size(), batch=args.batch,
        step_seconds=times, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        clinical_data_accessed=False, optimizer_updated=True, completed_at_utc=now()))
    dist.barrier()
    distributed.disable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        import os
        save(Path(args.output) / ('failure_' + os.environ.get('RANK', '0') + '.json'),
             dict(state='failed', out_of_memory=isinstance(exc, torch.OutOfMemoryError),
                  error_type=type(exc).__name__, clinical_data_accessed=False))
        raise SystemExit(1)
