"""Explicit global ranks across twenty separate one-GPU Slurm allocations."""
import os
import sys
from configuration import WORK,WORLD,SEED,ARMS


def main():
    rank=int(os.environ['POOL_RANK'])
    if not 0<=rank<WORLD:raise RuntimeError('invalid_global_pool_rank')
    os.environ.update(RANK=str(rank),WORLD_SIZE=str(WORLD),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',
        COMPARISON_RANK=str(rank),COMPARISON_WORLD=str(WORLD),COMPARISON_ARM=ARMS[0],
        COMPARISON_PHASE='training',COMPARISON_SEED=str(SEED))
    if sys.argv[1]=='probe':
        os.environ['MASTER_PORT']=str(int(os.environ['COMPARISON_PORT'])+1)
        from gpu_probe import main as probe
        probe()
    elif sys.argv[1]=='train':
        folder=WORK/ARMS[0]/'training'
        sys.argv=[str(WORK/'rank.py'),'--config-file',str(folder/'config.yaml'),
            '--output-dir',str(folder),'--seed',str(SEED)]
        from rank import main as train
        train()
    else:raise RuntimeError('invalid_pool_worker_mode')


if __name__=='__main__':main()
