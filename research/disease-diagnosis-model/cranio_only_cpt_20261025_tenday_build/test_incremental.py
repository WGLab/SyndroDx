"""Synthetic resize scheduling, exposure continuity and guard-boundary checks."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from configuration import WORK
from common import PYTHON,load,save
from incremental import choose_world,exposure,verified_boundary,snapshot
from continuation import loader_options
from dinov3.data.samplers import InfiniteSampler


class PolicyTests(unittest.TestCase):
    def test_start_without_waiting_for_twenty(self):
        self.assertEqual(choose_world(3,294400),0)
        self.assertEqual(choose_world(4,294400),4)
        self.assertEqual(choose_world(9,294400),8)
        self.assertEqual(choose_world(20,294400),20)
    def test_only_grow_and_only_representable_offsets(self):
        self.assertEqual(choose_world(7,294400,current=4),4)
        self.assertEqual(choose_world(9,294400,current=4),8)
        self.assertEqual(choose_world(20,294401,current=4),4)
    def test_multi_resize_sampler_preserves_original_rank_specific_stream(self):
        from canonical_sampler import CanonicalSampler
        from dinov3.data.samplers import ShardedInfiniteSampler
        count=320;consumed=80;seed=73
        lanes=[list(itertools.islice(iter(ShardedInfiniteSampler(sample_count=count,shuffle=True,
            seed=seed,start=rank,step=16)),700)) for rank in range(16)]
        expected=np.asarray(lanes).T.reshape(-1).tolist()
        with tempfile.TemporaryDirectory() as cache:
            for world in (4,5,8,10,16,20):
                pieces=[list(itertools.islice(iter(CanonicalSampler(sample_count=count,seed=seed,
                    rank=rank,world=world,consumed=consumed,cache=cache)),80)) for rank in range(world)]
                actual=np.asarray(pieces).T.reshape(-1).tolist()
                self.assertEqual(actual,expected[consumed:consumed+world*80])
                consumed+=world*80
    def test_resize_does_not_change_actual_learning_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for world in (4,8,20):
                path=Path(tmp)/f'{world}.json';save(path,dict(arm='synthetic',world_size=world,completed_steps=625,
                    source_world=8,source_training='/synthetic',consumed_global_samples=304000))
                env=dict(os.environ,POOL_ATTEMPT_PLAN=str(path),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                code="from configuration import *;import math;cfg=configuration(ARMS[0]);assert WORLD==%d;assert RESUME_STEP==625;assert SOURCE_WORLD==8;assert abs(cfg.schedules.lr.peak*4*math.sqrt(BATCH*WORLD/1024)-HEAD_LR)<1e-15"%world
                subprocess.run([str(PYTHON),'-c',code],env=env,check=True,capture_output=True,timeout=60)
    def test_negative_or_regressed_exposure_rejected(self):
        with self.assertRaises(RuntimeError):exposure(0,600,599,4)
        with self.assertRaises(RuntimeError):loader_options(dict(seed=5,sampler_advance=32,batch_size=32),1,target_world=4,consumed_global_samples=-4)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.folder=self.root/'training'
        self.folder.mkdir();(self.folder/'ckpt/599').mkdir(parents=True)
        (self.folder/'ckpt/599/.metadata').write_bytes(b'synthetic')
        (self.folder/'ckpt/599/shard').write_bytes(b'synthetic model optimizer and teacher')
        (self.folder/'config.yaml').write_text('synthetic: true\n')
        v=self.folder/'validation/step_000600';v.mkdir(parents=True)
        save(v/'report.json',dict(decision=dict(continue_training=True)))
        save(self.folder/'validation/monitor_state.json',dict(completed_steps=600,supported_declines=['synthetic_endpoint']))
        save(self.folder/'validation/last_accepted.json',dict(completed_steps=600))
        for rank in range(4):
            save(self.folder/f'rank_{rank:02d}_receipt.json',dict(success=True,state='paused_for_pool_expansion',completed_steps=600,consumed_global_samples=297600))
            save(self.folder/f'resume_rank_{rank:02d}.json',dict(state='passed'))
            np.save(v/f'rank_{rank:02d}.private.npy',np.ones((2,3),dtype=np.float32))
    def tearDown(self):self.tmp.cleanup()
    def test_rejects_failed_guard_or_missing_rank_or_checkpoint(self):
        verified_boundary(self.folder,4)
        p=self.folder/'validation/step_000600/report.json';save(p,dict(decision=dict(continue_training=False)))
        with self.assertRaises(RuntimeError):verified_boundary(self.folder,4)
        save(p,dict(decision=dict(continue_training=True)))
        (self.folder/'rank_03_receipt.json').unlink()
        with self.assertRaises(RuntimeError):verified_boundary(self.folder,4)
    def test_immutable_snapshot_preserves_checkpoint_guard_and_validation(self):
        target=snapshot(self.folder,self.root/'snapshot',600,4)
        self.assertEqual(load(target/'validation/monitor_state.json')['supported_declines'],['synthetic_endpoint'])
        original=self.folder/'ckpt/599/shard';copied=target/'ckpt/599/shard'
        self.assertEqual(original.stat().st_ino,copied.stat().st_ino)
        original.unlink();self.assertTrue(copied.is_file())
        self.assertTrue((target/'validation/step_000600/rank_03.private.npy').is_file())
    def test_recover_only_confirmed_exited_diagnostic_owner(self):
        import pool_work
        (self.root/'staging').mkdir()
        save(self.root/'staging/ledger.json',dict(tasks={'a':dict(state='claimed',owner='exited'),
            'b':dict(state='claimed',owner='live'),'c':dict(state='complete')}))
        with patch.object(pool_work,'WORK',self.root):pool_work.release_owner_after_exit('exited')
        tasks=load(self.root/'staging/ledger.json')['tasks']
        self.assertEqual(tasks['a']['state'],'pending');self.assertEqual(tasks['b']['owner'],'live')
        self.assertEqual(tasks['c']['state'],'complete')


if __name__=='__main__':unittest.main()
