"""Synthetic continuation and prospective-guard checks; no patient data."""
import ast
import itertools
import unittest
from unittest.mock import patch
import numpy as np
from omegaconf import OmegaConf
from configuration import WORK,WORLD,BATCH,ARMS,configuration
from continuation import loader_options
from guard import decide,KEYS
from dinov3.data.samplers import InfiniteSampler
from dinov3.train import train as official


def intervals(lo=0.,hi=0.):
    return {c:{m:{'paired_patient_bootstrap_95_interval':[lo,hi]} for m in ('ridge','retrieval')}
            for c in ('disease','structural')}


class ResumeTests(unittest.TestCase):
    def test_one_twenty_gpu_encoder(self):
        self.assertEqual(WORLD,20);self.assertEqual(BATCH,32);self.assertEqual(len(ARMS),1)
        self.assertEqual(configuration(ARMS[0]).train.batch_size_per_gpu*WORLD,640)

    def test_global_sample_offset_uses_old_exposure(self):
        original=dict(seed=20261019+575+1,sampler_advance=575*32,batch_size=32)
        result=loader_options(original,575)
        self.assertEqual(result['seed'],20261020)
        self.assertEqual(result['sampler_advance']*20,575*512)
        self.assertEqual(original['sampler_advance'],575*32)

    def test_bad_offset_fails(self):
        with self.assertRaises(RuntimeError):loader_options(dict(seed=1,sampler_advance=1,batch_size=32),575)

    def test_fractional_repartition_offset_fails(self):
        with self.assertRaises(RuntimeError):loader_options(dict(seed=1,sampler_advance=32,batch_size=32),1)

    def test_repartition_preserves_next_global_stream(self):
        complete=5;consumed=5*16;result=[]
        options=loader_options(dict(seed=8+complete,sampler_advance=complete*4,batch_size=4),complete,source_global_batch=16,target_world=5)
        for rank in range(5):
            sampler=InfiniteSampler(sample_count=1000,shuffle=True,seed=options['seed'],start=rank,step=5,advance=options['sampler_advance'])
            result.extend(itertools.islice(iter(sampler),4))
        whole=InfiniteSampler(sample_count=1000,shuffle=True,seed=8,start=0,step=1,advance=consumed)
        self.assertEqual(sorted(result),sorted(itertools.islice(iter(whole),20)))

    def test_actual_lr_schedules_not_scaled_up_with_world(self):
        old=OmegaConf.load(WORK/'source_snapshot/config.yaml');new=configuration(ARMS[0])
        with patch.object(official.distributed,'get_world_size',return_value=16):before=official.build_schedulers(old)
        with patch.object(official.distributed,'get_world_size',return_value=20):after=official.build_schedulers(new)
        for a,b in zip(before,after):np.testing.assert_allclose(a,b[:len(a)],rtol=1e-13,atol=1e-15)

    def test_script_syntax(self):
        for path in WORK.glob('*.py'):ast.parse(path.read_text())


class GuardTests(unittest.TestCase):
    def setUp(self):self.anchor=dict.fromkeys(KEYS,.5)
    def test_resume_anchor_passes_unchanged(self):
        self.assertTrue(decide(self.anchor,self.anchor,comparisons=intervals())['continue_training'])
    def test_uncertain_small_change_warns(self):
        result=decide(dict(self.anchor,disease_ridge_top1=.49),self.anchor,comparisons=intervals(-.1,.1))
        self.assertTrue(result['continue_training']);self.assertEqual(result['state'],'warning_validation_change')
    def test_further_ten_point_drop_stops(self):
        self.assertFalse(decide(dict(self.anchor,disease_ridge_top1=.39),self.anchor,comparisons=intervals(-.3,.1))['continue_training'])
    def test_supported_two_point_drop_stops(self):
        self.assertFalse(decide(dict(self.anchor,disease_ridge_top1=.47),self.anchor,comparisons=intervals(-.08,-.001))['continue_training'])
    def test_supported_persistent_small_drop_stops(self):
        values=dict(self.anchor,disease_ridge_top1=.495)
        first=decide(values,self.anchor,comparisons=intervals(-.01,-.001))
        self.assertTrue(first['continue_training'])
        self.assertFalse(decide(values,self.anchor,comparisons=intervals(-.01,-.001),previous_supported=first['supported_declines'])['continue_training'])
    def test_missing_interval_stops(self):
        self.assertFalse(decide(self.anchor,self.anchor,comparisons={})['continue_training'])
    def test_nonfinite_metric_stops(self):
        self.assertFalse(decide(dict(self.anchor,disease_ridge_top1=float('nan')),self.anchor,comparisons=intervals())['continue_training'])


if __name__=='__main__':unittest.main()
