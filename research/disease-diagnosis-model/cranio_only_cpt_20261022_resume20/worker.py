"""One Slurm task per GPU; all tasks participate in one shared encoder."""
import os
import sys
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK))
from configuration import WORLD,SEED,ARMS


def main():
    rank=int(os.environ['SLURM_PROCID'])
    if int(os.environ['SLURM_NTASKS'])!=WORLD:raise RuntimeError('slurm_world_size_mismatch')
    os.environ.update(RANK=str(rank),WORLD_SIZE=str(WORLD),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',
        COMPARISON_RANK=str(rank),COMPARISON_WORLD=str(WORLD),COMPARISON_ARM=ARMS[0],
        COMPARISON_PHASE='training',COMPARISON_SEED=str(SEED))
    mode=sys.argv[1]
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('exactly_one_bound_gpu_required_per_task')
    if mode=='probe':
        os.environ['MASTER_PORT']=str(int(os.environ['COMPARISON_PORT'])+1)
        from gpu_probe import main as probe
        probe()
    elif mode=='train':
        folder=WORK/ARMS[0]/'training'
        sys.argv=[str(WORK/'rank.py'),'--config-file',str(folder/'config.yaml'),
            '--output-dir',str(folder),'--seed',str(SEED)]
        from rank import main as train
        train()
    else:raise RuntimeError('unrecognized_worker_mode')


if __name__=='__main__':main()
