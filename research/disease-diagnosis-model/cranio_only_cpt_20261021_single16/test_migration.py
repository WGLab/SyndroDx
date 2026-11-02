"""Synthetic checks for the single-model, constant-global-batch migration."""
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from omegaconf import OmegaConf
from configuration import WORK,ARMS,SEEDS,WORLD,BATCH,configuration,RESUME_STEP
from allocation_pool import select_pools
from continuation import compare_resharded_features,loader_options
from dinov3.data.samplers import InfiniteSampler
from dinov3.train import train as official


class MigrationTests(unittest.TestCase):
    def test_one_model_16_ranks_global_batch512(self):
        self.assertEqual(len(ARMS),1);self.assertEqual(WORLD,16)
        self.assertEqual(BATCH,32);self.assertEqual(WORLD*BATCH,512)
        self.assertEqual(configuration(ARMS[0]).train.seed,20261019)

    def test_full_schedule_unchanged_with_constant_global_batch(self):
        old=OmegaConf.load(WORK.with_name('cranio_only_cpt_20261021_continue1000')/'cranio_seed19/training/config.yaml')
        new=configuration(ARMS[0])
        with patch.object(official.distributed,'get_world_size',return_value=8):a=official.build_schedulers(old)
        with patch.object(official.distributed,'get_world_size',return_value=16):b=official.build_schedulers(new)
        for left,right in zip(a,b):self.assertTrue(np.array_equal(left,right))

    def test_next_global_batch_indices_preserved_8_to16(self):
        # A full period is divisible by both worlds, just as the real virtual dataset.
        count=4096;completed=3
        def batch(world,local):
            result=[]
            options=loader_options(dict(seed=123+completed,sampler_advance=completed*local,batch_size=local),completed)
            for rank in range(world):
                sampler=InfiniteSampler(sample_count=count,shuffle=True,seed=options['seed'],
                    start=rank,step=world,advance=options['sampler_advance'])
                result.extend(itertools.islice(iter(sampler),local))
            return sorted(result)
        self.assertEqual(batch(8,64),batch(16,32))

    def test_validation_feature_repartition(self):
        with tempfile.TemporaryDirectory() as name:
            source=Path(name)/'source';target=Path(name)/'target'
            source.mkdir();target.mkdir()
            features=np.eye(37,dtype=np.float32)
            for world,folder in [(8,source),(16,target)]:
                for rank in range(world):np.save(folder/f'rank_{rank:02d}.private.npy',features[rank::world])
            self.assertEqual(compare_resharded_features(source,target,8,16),1.)
            np.save(target/'rank_00.private.npy',features[:1])
            with self.assertRaises(RuntimeError):compare_resharded_features(source,target,8,16)

    def fixture(self):
        specs={ARMS[0]:[]};records={};steps=[]
        for rank in range(WORLD):
            job=str(100+rank);node='node'+str(rank//8)
            specs[ARMS[0]].append(dict(job_id=job,lookup_id=job,node=node,host_memory_gib=68,allowed_idle_steps={job+'.0':'sleep'}))
            records[job]=dict(JobId=job,UserId='research_user(1)',JobState='RUNNING',NodeList=node,
                Partition='gpu-xe9680q',AllocTRES='cpu=4,mem=72G,gres/gpu=1')
            steps.append((job+'.0','sleep'))
        return specs,records,steps

    def test_two_nodes_and_all_sixteen_allocations_admitted(self):
        s,r,t=self.fixture();self.assertEqual(len(select_pools(s,r,t)[ARMS[0]]),16)

    def test_unknown_worker_blocks_reuse(self):
        s,r,t=self.fixture();t.append(('100.9','python'))
        self.assertEqual(len(select_pools(s,r,t)[ARMS[0]]),15)

    def test_exact_own_coordinator_only(self):
        s,r,t=self.fixture();t.append(('100.9','cranio-single16-coordinator'))
        with patch.dict('os.environ',{'SLURM_JOB_ID':'100','SLURM_STEP_ID':'9'}):
            self.assertEqual(len(select_pools(s,r,t)[ARMS[0]]),16)
        self.assertEqual(len(select_pools(s,r,t)[ARMS[0]]),15)

    def test_duplicate_owner_and_wrong_topology_rejected(self):
        s,r,t=self.fixture();r['100']['UserId']='other(2)'
        with self.assertRaises(RuntimeError):select_pools(s,r,t)
        s,r,t=self.fixture();s[ARMS[0]][1]['job_id']='100'
        with self.assertRaises(RuntimeError):select_pools(s,r,t)
        s,r,t=self.fixture();s[ARMS[0]][0]['node']='third-node'
        with self.assertRaises(RuntimeError):select_pools(s,r,t)


if __name__=='__main__':unittest.main()
