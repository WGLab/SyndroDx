"""Synthetic checks of loss gradients, identical exposure and evaluation isolation."""
import unittest,importlib.util,tempfile,os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matched_runtime import *
from matched_models import dino_loss,update_teacher
from matched_data import MatchedFaces,TwoViews
from data import StreamSampler

class Pixels:
    def image(self,sha):return Image.fromarray(np.arange(140*150*3,dtype=np.uint8).reshape(140,150,3))

class Checks(unittest.TestCase):
    def test_official_dino_loss_and_gradient(self):
        spec=importlib.util.spec_from_file_location('original_dino',SSL/'vendor/dino_loss.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        official=m.DINOLoss(17,2,.04,.07,1000,4001)
        # Original center collective tested separately; this check isolates exact objective/gradient.
        official.update_center=lambda x:None
        torch.manual_seed(21);x=torch.randn(16,17,requires_grad=True);t=torch.randn(16,17);center=torch.randn(1,17)*.01
        official.center.copy_(center);loss=official(x,t,357);g=torch.autograd.grad(loss,x)[0]
        xx=x.detach().clone().requires_grad_(True);temp=float(official.teacher_temp_schedule[357]);mine=dino_loss(xx,t,center,temp)
        gg=torch.autograd.grad(mine,xx)[0]
        torch.testing.assert_close(loss,mine);torch.testing.assert_close(g,gg)
    def test_microbatch_weighting(self):
        torch.manual_seed(1);x=torch.randn(26,17,requires_grad=True);t=torch.randn(26,17);center=torch.zeros(1,17)
        full=dino_loss(x,t,center,.07);g=torch.autograd.grad(full,x)[0]
        xx=x.detach().clone().requires_grad_(True);acc=0
        for lo,hi in [(0,4),(4,9),(9,13)]:
            ix=list(range(lo,hi))+list(range(13+lo,13+hi))
            acc=acc+dino_loss(xx[ix],t[ix],center,.07)*(hi-lo)/13
        torch.testing.assert_close(full,acc);torch.testing.assert_close(g,torch.autograd.grad(acc,xx)[0])
    def test_world_resize_preserves_draws(self):
        for world in [1,2,3,4,5,6,8,10,12,15,16,20]:
            batch=3840//world;ids=[]
            for rank in range(world):
                it=iter(StreamSampler(3840*250,rank,world,batch));ids.extend(next(it) for _ in range(batch))
            self.assertEqual(sorted(ids),list(range(3840*250,3840*251)))
    def test_same_views_across_arm_and_rng(self):
        ds=MatchedFaces.__new__(MatchedFaces);ds.groups=(((('synthetic','abc'),),),);ds.packed=Pixels();ds.transform=TwoViews()
        torch.manual_seed(1);first,y=ds[42]
        torch.manual_seed(912);second,z=ds[42]
        self.assertEqual(y,z)
        for a,b in zip(first,second):torch.testing.assert_close(a,b)
        self.assertFalse(torch.equal(first[0],first[1]))
    def test_ema(self):
        a=torch.nn.Linear(3,2);b=__import__('copy').deepcopy(a);before=b.weight.detach().clone()
        with torch.no_grad():a.weight.add_(1)
        update_teacher(a,b,.9);torch.testing.assert_close(b.weight,before+.1)
    def test_hpo_missing_is_not_negative(self):
        from unittest.mock import patch
        from score import hpo_score
        term='HP:synthetic'
        positive=dict(hpo=[term],all_present_hpo=[term],absent_hpo=[])
        unknown=dict(hpo=[],all_present_hpo=[],absent_hpo=[])
        negative=dict(hpo=[],all_present_hpo=[],absent_hpo=[term])
        ontology=dict(facial_roots=[term],terms={term:dict(parents=[])})
        rng=np.random.default_rng(3);x=rng.normal(size=(20,3));q=rng.normal(size=(8,3))
        with patch('score.load',return_value=ontology):
            no_negative=hpo_score(x,q,[positive]*10+[unknown]*10,[positive]*4+[unknown]*4)
            self.assertEqual(no_negative['state'],'not_evaluable')
            with_negative=hpo_score(x,q,[positive]*10+[negative]*5+[unknown]*5,[positive]*2+[negative]*2+[unknown]*4)
            self.assertEqual(with_negative['eligible_terms'],1)
            self.assertEqual(with_negative['known_assertion_pairs'],4)
    def test_fewshot_excludes_unsupported_query_classes(self):
        from score import fewshot
        x=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],dtype=np.float32)
        q=np.array([[1.,0.],[0.,1.],[-1.,0.]],dtype=np.float32)
        r=fewshot(x,q,['a','a','b','b'],['a','b','unsupported'])
        self.assertEqual(r['1']['query_patients'],2)
        self.assertEqual(r['1']['macro_top1'],1.)
        self.assertEqual(r['5']['state'],'not_evaluable')

if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    save(ROOT/'tests.json',dict(state='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),synthetic_only=True,updated_at_utc=now()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
