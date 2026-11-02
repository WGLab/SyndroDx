"""Synthetic tests: angular margin, class association, splits and restart."""
import itertools
import collections
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from PIL import Image
from model import IdentityHead,fixed_batchnorm
from vendor.author_margin_losses import ArcFace
from data import ClinicalFaces,StreamSampler,validate_split_rows
from train import set_learning_rates

class IdentityTests(unittest.TestCase):
    def test_margin_changes_only_target_angle(self):
        c=torch.tensor([[.6,.2,-.1],[-.2,.7,.1]],requires_grad=True)
        target=torch.tensor([0,1]);loss=ArcFace(s=64.,margin=.5)
        actual=loss(c.clone(),target)
        expected=c.detach().clone()
        expected[0,0]=math.cos(math.acos(.6)+.5)
        expected[1,1]=math.cos(math.acos(.7)+.5)
        torch.testing.assert_close(actual,expected*64.)
        F.cross_entropy(actual,target).backward()
        self.assertTrue(torch.isfinite(c.grad).all())

    def test_arcface_matches_angular_reference_gradient(self):
        c=torch.tensor([[.6,.2,-.1],[-.2,.7,.1]],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([0,1]);actual=ArcFace(s=4.,margin=.3)(c.clone(),target)
        ga=torch.autograd.grad(F.cross_entropy(actual,target),c)[0]
        v=c.detach().clone().requires_grad_()
        mask=F.one_hot(target,3).bool()
        reference=torch.where(mask,torch.cos(torch.acos(v)+.3),v)*4.
        gr=torch.autograd.grad(F.cross_entropy(reference,target),v)[0]
        torch.testing.assert_close(ga,gr)

    def test_head_normalization_and_boundary_are_finite(self):
        head=IdentityHead(4,feature_dim=4)
        with torch.no_grad():head.weight.copy_(torch.eye(4))
        x=torch.eye(4,requires_grad=True);target=torch.arange(4)
        logits=head(x,target)
        torch.testing.assert_close(logits,head(x*7,target))
        F.cross_entropy(logits,target).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_stream_world_resize_has_disjoint_exposure_indices(self):
        a=[list(itertools.islice(iter(StreamSampler(0,r,3,8)),16)) for r in range(3)]
        self.assertEqual(sorted(sum(a,[])),list(range(48)))
        b=[list(itertools.islice(iter(StreamSampler(48,r,5,8)),8)) for r in range(5)]
        self.assertEqual(sorted(sum(b,[])),list(range(48,88)))

    def test_sample_image_remains_associated_with_identity_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.png';b=Path(tmp)/'b.png'
            Image.new('RGB',(8,8),(10,0,0)).save(a);Image.new('RGB',(8,8),(20,0,0)).save(b)
            dataset=ClinicalFaces.__new__(ClinicalFaces)
            dataset.groups=((((str(a),'a'),),),(((str(b),'b'),),))
            dataset.transform=lambda im:im.getpixel((0,0))[0];dataset.checked={str(a),str(b)}
            dataset.cache_hashes={'a':hashlib.sha256(a.read_bytes()).hexdigest(),'b':hashlib.sha256(b.read_bytes()).hexdigest()}
            dataset.encoded_cache=collections.OrderedDict()
            with patch('data.cache_path',side_effect=lambda sha,kind:a if sha=='a' else b):
                observed=set()
                for index in range(100):
                    value,label=dataset[index]
                    self.assertEqual(value,10*(label+1));observed.add(label)
                self.assertEqual(observed,{0,1})

    def test_fast_cached_read_checks_pinned_hash_and_rejects_bad_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.png';path.write_bytes(b'corrupted-cache')
            dataset=ClinicalFaces.__new__(ClinicalFaces)
            dataset.cache_hashes={'synthetic':hashlib.sha256(b'correct-cache').hexdigest()}
            dataset.encoded_cache=collections.OrderedDict()
            with patch('data.cache_path',return_value=path),patch('data.verified_cached_path'):
                with self.assertRaisesRegex(RuntimeError,'cache_sha256_mismatch'):
                    dataset.image_bytes('synthetic-source','synthetic')

    def test_fast_cached_read_uses_verified_immutable_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.png';raw=b'synthetic-encoded-image';path.write_bytes(raw)
            dataset=ClinicalFaces.__new__(ClinicalFaces)
            dataset.cache_hashes={'synthetic':hashlib.sha256(raw).hexdigest()}
            dataset.encoded_cache=collections.OrderedDict()
            with patch('data.cache_path',return_value=path):
                self.assertEqual(dataset.image_bytes('source','synthetic'),raw)
                path.unlink()
                self.assertEqual(dataset.image_bytes('source','synthetic'),raw)

    def test_patient_or_pixel_leakage_rejected(self):
        train=[dict(group_id='synthetic_a',pixel_sha256='x',source='approved_internal',split='train',training_admitted='true',view='profile')]
        with self.assertRaisesRegex(RuntimeError,'patient_overlap'):
            validate_split_rows(train,[dict(group_id='synthetic_a',pixel_sha256='y')],[])
        with self.assertRaisesRegex(RuntimeError,'identical_pixels'):
            validate_split_rows(train,[dict(group_id='synthetic_b',pixel_sha256='x')],[])

    def test_new_head_warmup_and_backbone_ramp(self):
        class Opt: param_groups=[dict(base_lr=2e-6,family='backbone'),dict(base_lr=.001,family='head')]
        p=dict(head_only_steps=100,backbone_ramp_steps=500);opt=Opt()
        set_learning_rates(opt,99,p);self.assertEqual(opt.param_groups[0]['lr'],0.)
        set_learning_rates(opt,100,p);self.assertAlmostEqual(opt.param_groups[0]['lr'],4e-9)
        set_learning_rates(opt,599,p);self.assertEqual(opt.param_groups[0]['lr'],2e-6)

    def test_optimizer_resume_reproduces_next_update(self):
        torch.manual_seed(4);model=IdentityHead(5,feature_dim=4);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(8,4);labels=torch.arange(8)%5
        F.cross_entropy(model(x,labels),labels).backward();opt.step();opt.zero_grad()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path)
            state=torch.load(path,weights_only=True)
            restored=IdentityHead(5,feature_dim=4);restored.load_state_dict(state['model'])
            newopt=torch.optim.AdamW(restored.parameters(),lr=9.);newopt.load_state_dict(state['optimizer'])
            for net,optimizer in ((model,opt),(restored,newopt)):
                F.cross_entropy(net(x,labels),labels).backward();optimizer.step()
            torch.testing.assert_close(model.weight,restored.weight)

    def test_fixed_batchnorm_keeps_affine_trainable(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4)).train()
        fixed_batchnorm(model);old=model[1].running_mean.clone()
        model(torch.randn(8,4)).square().mean().backward()
        self.assertTrue(torch.equal(model[1].running_mean,old))
        self.assertIsNotNone(model[1].weight.grad)

if __name__=='__main__':unittest.main()
