"""Synthetic scientific-integrity and resume tests; no clinical I/O."""
import itertools
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from data import StreamSampler,validate_split_rows
from model import update_teacher,fixed_batchnorm
from vendor.dino_loss import DINOLoss
from train import set_learning_rates

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        dist.init_process_group('gloo',init_method='file://'+cls.temp.name+'/dist',rank=0,world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group();cls.temp.cleanup()

    def test_stream_has_no_duplicate_exposure_after_world_resize(self):
        a=[list(itertools.islice(iter(StreamSampler(0,r,3,8)),16)) for r in range(3)]
        self.assertEqual(sorted(sum(a,[])),list(range(48)))
        b=[list(itertools.islice(iter(StreamSampler(48,r,5,8)),8)) for r in range(5)]
        self.assertEqual(sorted(sum(b,[])),list(range(48,88)))

    def test_patient_and_pixel_leakage_rejected(self):
        train=[dict(group_id='synthetic_a',pixel_sha256='x',source='approved_internal',split='train',training_admitted='true',view='profile')]
        with self.assertRaisesRegex(RuntimeError,'patient_overlap'):
            validate_split_rows(train,[dict(group_id='synthetic_a',pixel_sha256='y')],[])
        with self.assertRaisesRegex(RuntimeError,'identical_pixels'):
            validate_split_rows(train,[dict(group_id='synthetic_b',pixel_sha256='x')],[])
        self.assertTrue(validate_split_rows(train,[dict(group_id='synthetic_b',pixel_sha256='y')],[])['patient_disjoint'])

    def test_official_loss_cross_view_terms_and_teacher_stop_gradient(self):
        torch.manual_seed(12)
        student=torch.randn(8,11,requires_grad=True);teacher=torch.randn(8,11,requires_grad=True)
        loss=DINOLoss(11,2,.04,.07,3,8)
        value=loss(student,teacher,0)
        q=torch.softmax(teacher.detach()/.04,-1).chunk(2)
        logp=torch.log_softmax(student/.1,-1).chunk(2)
        expected=(-(q[0]*logp[1]).sum(-1).mean()-(q[1]*logp[0]).sum(-1).mean())/2
        self.assertTrue(torch.allclose(value,expected))
        self.assertTrue(torch.allclose(loss.center,teacher.detach().mean(0,keepdim=True)*.1))
        value.backward()
        self.assertIsNone(teacher.grad);self.assertIsNotNone(student.grad)

    def test_ema_updates_parameters_and_copies_bn_buffers(self):
        student=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        teacher=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        for p in student.parameters():p.data.fill_(2.)
        for p in teacher.parameters():p.data.fill_(0.)
        student[1].running_mean.fill_(7.)
        update_teacher(student,teacher,.75)
        self.assertTrue(all(torch.equal(p,torch.full_like(p,.5)) for p in teacher.parameters()))
        self.assertTrue(torch.equal(teacher[1].running_mean,student[1].running_mean))

    def test_bn_affine_gradients_survive_fixed_population_statistics(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4)).train()
        fixed_batchnorm(model);old=model[1].running_mean.clone()
        model(torch.randn(8,4)).square().mean().backward()
        self.assertTrue(torch.equal(model[1].running_mean,old))
        self.assertIsNotNone(model[1].weight.grad)

    def test_checkpoint_resume_retains_optimizer_and_loss_center(self):
        student=torch.nn.Linear(4,4);optimizer=torch.optim.AdamW(student.parameters(),lr=.01)
        student(torch.ones(2,4)).square().mean().backward();optimizer.step()
        loss=DINOLoss(4,2,.04,.07,2,10);loss.center.fill_(3.)
        path=Path(self.temp.name)/'checkpoint.pt'
        torch.save(dict(model=student.state_dict(),optimizer=optimizer.state_dict(),loss=loss.state_dict()),path)
        state=torch.load(path,weights_only=True)
        restored=torch.nn.Linear(4,4);restored.load_state_dict(state['model'])
        opt=torch.optim.AdamW(restored.parameters(),lr=9.);opt.load_state_dict(state['optimizer'])
        newloss=DINOLoss(4,2,.04,.07,2,10);newloss.load_state_dict(state['loss'])
        self.assertEqual(opt.param_groups[0]['lr'],.01)
        self.assertEqual(float(newloss.center.mean()),3.)
        for a,b in zip(student.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))

    def test_head_warmup_then_backbone_ramp(self):
        class Optimizer:
            param_groups=[dict(base_lr=1e-5,family='backbone'),dict(base_lr=1e-4,family='head')]
        opt=Optimizer();p=dict(head_only_steps=100,backbone_ramp_steps=100)
        set_learning_rates(opt,99,p);self.assertEqual(opt.param_groups[0]['lr'],0)
        set_learning_rates(opt,100,p);self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-7)
        set_learning_rates(opt,199,p);self.assertEqual(opt.param_groups[0]['lr'],1e-5)

if __name__=='__main__':unittest.main()
