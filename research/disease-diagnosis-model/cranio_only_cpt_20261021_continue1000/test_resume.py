"""Synthetic full optimizer/model resume, schedule and sampler-continuation checks."""
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd
from configuration import WORK,SOURCE_WORK,ARMS,WORLD,configuration,RESUME_STEP
from continuation import loader_options,restore_full_state
from dinov3.train import train as official
from dinov3.data.samplers import InfiniteSampler
from omegaconf import OmegaConf


class ResumeTests(unittest.TestCase):
    def test_sampler_continues_original_permutation(self):
        options=dict(seed=116,sampler_advance=12,batch_size=4)
        adjusted=loader_options(options,3)
        self.assertEqual(adjusted['seed'],113);self.assertEqual(options['seed'],116)
        original=InfiniteSampler(sample_count=29,shuffle=True,seed=113,start=1,step=4)
        resumed=InfiniteSampler(sample_count=29,shuffle=True,seed=adjusted['seed'],start=1,step=4,advance=12)
        self.assertEqual(list(itertools.islice(iter(original),12,32)),list(itertools.islice(iter(resumed),20)))
    def test_wrong_sampler_offset_fails(self):
        with self.assertRaises(RuntimeError):loader_options(dict(seed=116,sampler_advance=0,batch_size=4),3)
    def test_extension_preserves_first200_schedule_values(self):
        old=OmegaConf.load(SOURCE_WORK/ARMS[0]/'training/config.yaml');new=configuration(ARMS[0])
        with patch.object(official.distributed,'get_world_size',return_value=WORLD):
            a,b=official.build_schedulers(old),official.build_schedulers(new)
        for left,right in zip(a,b):
            self.assertTrue((left[:RESUME_STEP]==right[:RESUME_STEP]).all())
    def test_adamw_and_model_roundtrip_preserves_next_update(self):
        source=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(source.parameters(),lr=.003)
        x=torch.tensor([[1.,2.,3.],[2.,-1.,.5]])
        def step(model,opt):
            opt.zero_grad();model(x).square().mean().backward();opt.step()
        for _ in range(200):step(source,optimizer)
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'199'
            dcp.save(dict(iteration=199,model=dcpsd.get_model_state_dict(source),optimizer=dcpsd.get_optimizer_state_dict(source,optimizer)),checkpoint_id=p)
            target=torch.nn.Linear(3,2);restored=torch.optim.AdamW(target.parameters(),lr=.003)
            iteration,proof=restore_full_state(p,model=target,optimizer=restored,loader=official.load_checkpoint,completed_steps=200)
            self.assertEqual(iteration,199);self.assertEqual(proof['maximum_optimizer_step'],200)
            for left,right in zip(source.parameters(),target.parameters()):self.assertTrue(torch.equal(left,right))
            step(source,optimizer);step(target,restored)
            for left,right in zip(source.parameters(),target.parameters()):self.assertTrue(torch.equal(left,right))
    def test_missing_optimizer_is_rejected(self):
        with self.assertRaises(RuntimeError):restore_full_state(None,model=None,optimizer=None,loader=None,completed_steps=200)


if __name__=='__main__':unittest.main()
