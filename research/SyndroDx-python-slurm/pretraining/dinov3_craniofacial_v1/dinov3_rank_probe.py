"""Synthetic NCCL probe: no dataset or pretrained checkpoint is opened."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-address', required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    if torch.cuda.device_count() != 1:
        raise RuntimeError('expected_one_allocated_GPU_per_rank')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://{args.master_address}:{args.master_port}',
                            rank=args.rank, world_size=args.world_size,
                            timeout=datetime.timedelta(seconds=120))
    properties = torch.cuda.get_device_properties(0)
    tensor = torch.empty(8 * 1024 * 1024, device='cuda', dtype=torch.float32)
    expected = args.world_size * (args.world_size + 1) // 2
    for _ in range(5):
        tensor.fill_(args.rank + 1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.monotonic()
    for _ in range(50):
        tensor.fill_(args.rank + 1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    if not bool(torch.all(tensor == expected)):
        raise RuntimeError('collective_result_mismatch')
    result = {'state': 'collectives_passed', 'rank': args.rank, 'world_size': args.world_size,
              'host': socket.gethostname(), 'GPU_name': properties.name,
              'GPU_memory_GiB': round(properties.total_memory / 2**30, 2),
              'all_reduce_iterations': 50, 'payload_MiB': 32, 'elapsed_seconds': elapsed,
              'clinical_data_accessed': False, 'pretrained_weights_accessed': False}
    root = Path(args.output)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / f'rank_{args.rank:02d}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, target)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
