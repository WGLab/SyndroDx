"""Synthetic checks for pooling, weighted accumulation and retention metrics."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from data import macro_hits,retrieval,fewshot


class TrialTests(unittest.TestCase):
    def test_weighted_microbatch_gradient_matches_full_batch(self):
        torch.manual_seed(11);x=torch.randn(9,4);y=torch.tensor([0,0,1,2,1,1,2,0,2]);w=torch.tensor([.2,.7,1.5])
        weight=torch.randn(4,3,requires_grad=True)
        F.cross_entropy(x@weight,y,weight=w).backward();reference=weight.grad.clone();weight.grad=None
        for start in range(0,len(x),4):
            loss=F.cross_entropy(x[start:start+4]@weight,y[start:start+4],weight=w,reduction='sum')/w[y].sum()
            loss.backward()
        torch.testing.assert_close(reference,weight.grad)

    def test_multiview_pooling_keeps_patient_boundary(self):
        from model import JointModel
        m=JointModel.__new__(JointModel);torch.nn.Module.__init__(m)
        m.vision=torch.nn.Identity()
        x=torch.randn(5,512,requires_grad=True);xnorm=F.normalize(x,dim=1)
        actual=m.patient_features(xnorm,torch.tensor([2,3]),chunk=2)
        expected=torch.stack([F.normalize(xnorm[:2].mean(0),dim=0),F.normalize(xnorm[2:].mean(0),dim=0)])
        torch.testing.assert_close(actual,expected)
        actual[:,0].sum().backward();self.assertTrue(bool(x.grad.abs().sum()>0))

    def test_macro_weights_diseases_not_frequency(self):
        scores=np.array([[1,0],[1,0],[0,1]],float);labels=np.array([0,0,0])
        metrics,_=macro_hits(scores,labels);self.assertAlmostEqual(metrics['top1'],2/3)

    def test_retrieval_and_fewshot_use_training_support(self):
        z=np.array([[1,0],[0,1],[1,0],[0,1]],np.float32);labels=np.array([0,1,0,1])
        metrics,_=retrieval(z,np.array([0,1]),np.array([2,3]),labels)
        self.assertEqual(metrics['top1'],1.)
        f=fewshot(z,np.array([0,1]),np.array([2,3]),labels)
        self.assertEqual(f['1']['macro_top1_mean'],1.);self.assertEqual(f['5']['eligible_queries'],0)


if __name__=='__main__':unittest.main()
