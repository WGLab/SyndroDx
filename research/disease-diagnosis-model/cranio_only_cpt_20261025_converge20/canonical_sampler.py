"""Repartition the original 16-rank shuffled index stream without changing it.

Private index banks contain only synthetic dataset indices and remain local.
The original official sampler shuffles within rank-specific lanes. Reusing its
seed with a different world size would change these lanes, even with the right
sample count. Keep the original lanes and repartition their interleaved stream.
"""
import fcntl
import itertools
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Sampler
from dinov3.data.samplers import ShardedInfiniteSampler


class CanonicalSampler(Sampler):
    def __init__(self,*,sample_count,seed,rank,world,consumed,cache,source_world=16):
        if sample_count%source_world:raise RuntimeError('canonical_source_lanes_must_have_equal_length')
        if consumed%world:raise RuntimeError('nonintegral_global_sample_offset')
        self.sample_count=sample_count;self.seed=seed;self.rank=rank;self.world=world
        self.consumed=consumed;self.cache=Path(cache);self.source_world=source_world
    def bank(self,epoch):
        self.cache.mkdir(parents=True,exist_ok=True)
        path=self.cache/f'n{self.sample_count}_seed{self.seed}_source{self.source_world}_epoch{epoch}.private.npy'
        with path.with_suffix('.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            if not path.exists():
                lanes=[]
                for rank in range(self.source_world):
                    sampler=ShardedInfiniteSampler(sample_count=self.sample_count,shuffle=True,seed=self.seed,
                        start=rank,step=self.source_world,advance=0)
                    # Pinned upstream iteration counter seeds exactly this epoch.
                    sampler._iter_count=epoch
                    lanes.append(np.fromiter(itertools.islice(iter(sampler),self.sample_count//self.source_world),dtype=np.int32))
                values=np.stack(lanes,axis=1).reshape(-1)
                temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
                with temporary.open('wb') as out:np.save(out,values,allow_pickle=False)
                temporary.replace(path)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    def __iter__(self):
        position=self.consumed+self.rank
        while True:
            epoch,offset=divmod(position,self.sample_count)
            bank=self.bank(epoch)
            for value in bank[offset::self.world]:yield int(value)
            count=(self.sample_count-1-offset)//self.world+1
            position+=count*self.world
