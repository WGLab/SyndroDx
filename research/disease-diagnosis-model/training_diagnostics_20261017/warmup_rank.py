"""Local dataset and diagnostics adapters; official FINO optimizer loop unchanged."""
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
WORK = Path(__file__).resolve().parent
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
VENDOR = WORK / 'vendor/dinov3_fino'
sys.path[:0] = [str(VENDOR), str(BASE), str(V1)]
from common import load, save, now, install_rope_dtype_compatibility
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
from dinov3.train import train as official
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dataset import make_dataset as released_dataset


class PilotFinished(Exception):
    pass


class SyntheticDataset(Dataset):
    def __init__(self, transform, target_transform):
        self.transform, self.target_transform = transform, target_transform
    def __len__(self):
        return 8192
    def __getitem__(self, index):
        image = Image.new('RGB', (192, 224), (index % 255, index * 7 % 255, index * 19 % 255))
        return self.transform(image), self.target_transform(0) if self.target_transform else 0


def registered_dataset(*, dataset_str, transform, target_transform=None, **kwargs):
    if os.environ['WARMUP_PHASE'] == 'capacity':
        return SyntheticDataset(transform, target_transform)
    return released_dataset(dataset_str=dataset_str, transform=transform, target_transform=target_transform, **kwargs)


def fingerprint(module):
    result = hashlib.sha256()
    for parameter in module.parameters():
        value = parameter.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        result.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    rank, world = int(os.environ['WARMUP_RANK']), int(os.environ['WARMUP_WORLD'])
    phase = os.environ['WARMUP_PHASE']
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                     MASTER_ADDR=os.environ['WARMUP_MASTER'], MASTER_PORT=os.environ['WARMUP_PORT'],
                     TORCHELASTIC_RUN_ID='frozen-head-pilot-'+os.environ['WARMUP_PORT'])
    os.umask(0o077)
    torch.set_num_threads(4)
    install_rope_dtype_compatibility()
    folder = WORK / 'warmup_pilot' / phase
    original_init, original_checkpoint = SSLMetaArch.init_weights, official.save_checkpoint
    original_losses = SSLMetaArch.compute_losses
    reference = torch.load(BASE / 'initialization/public_synthetic_reference.pt', map_location='cpu', weights_only=True)
    model_ref = {}

    def checked_init(self):
        original_init(self)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            actual = self.teacher.backbone(reference['input'].cuda(), is_training=True)['x_norm_clstoken'].float()
        cosine = torch.nn.functional.cosine_similarity(actual, reference['expected_cls'].cuda()).min()
        dist.all_reduce(cosine, op=dist.ReduceOp.MIN)
        if not bool(torch.isfinite(cosine)) or float(cosine) < .999:
            raise RuntimeError('original_initialization_equivalence_failed')
        model_ref['model'] = self
        model_ref['fingerprint'] = fingerprint(self.student.backbone)
        torch.cuda.reset_peak_memory_stats()

    def checked_checkpoint(ckpt_dir, *args, **kwargs):
        original_checkpoint(ckpt_dir, *args, **kwargs)
        model = model_ref['model']
        same = fingerprint(model.student.backbone) == model_ref['fingerprint']
        flag = torch.tensor(int(same), device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not bool(flag):
            raise RuntimeError('frozen_backbone_changed')
        step = int(Path(ckpt_dir).name) + 1
        if rank == 0:
            save(folder / 'checkpoint_status.json', {'state':'checkpoint_verified', 'completed_steps':step,
                  'all_student_backbone_shards_bitwise_unchanged':True, 'updated_at_utc':now()})
        if phase == 'training' and step >= int(os.environ.get('WARMUP_STOP_AFTER', '200')):
            dist.barrier()
            raise PilotFinished()

    def measured_losses(self, *args, **kwargs):
        loss, metrics = original_losses(self, *args, **kwargs)
        if kwargs['iteration'] % 10 == 0:
            with torch.no_grad():
                p = kwargs['teacher_global']['cls_centered'].flatten(0, 1).float()
                entropy_sum = -(p * p.clamp_min(1e-30).log()).sum()
                count = torch.tensor(float(len(p)), device=p.device)
                marginal = p.sum(0)
                dist.all_reduce(entropy_sum);dist.all_reduce(count);dist.all_reduce(marginal)
                marginal /= count
                conditional = entropy_sum / count
                information = -(marginal * marginal.clamp_min(1e-30).log()).sum() - conditional
                metrics['dino_teacher_entropy'] = conditional
                metrics['dino_teacher_information'] = information
        return loss, metrics

    official.make_dataset = registered_dataset
    official.save_checkpoint = checked_checkpoint
    SSLMetaArch.init_weights = checked_init
    SSLMetaArch.compute_losses = measured_losses
    start, success, error = time.monotonic(), False, None
    try:
        official.main()
        success = True
    except PilotFinished:
        success = True
    except Exception as failure:
        error = failure
        raise
    finally:
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            save(folder / f'rank_{rank}_receipt.json', {'state':'passed' if success else 'failed',
                 'rank':rank, 'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                 'CUDA_free_GiB':free/2**30, 'elapsed_seconds':time.monotonic()-start,
                 'error_type':type(error).__name__ if error else None,
                 'clinical_data_accessed':phase != 'capacity', 'completed_at_utc':now()})
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
