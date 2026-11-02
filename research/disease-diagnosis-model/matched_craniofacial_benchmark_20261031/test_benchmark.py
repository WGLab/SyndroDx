"""Synthetic-only checks of matching, encoder structure and paired statistics."""
import sys,unittest,random
import numpy as np,torch
from common import *
from data import Images,GestaltMatcherDataset
from network import Encoder

class Checks(unittest.TestCase):
    def test_canonical_RGB(self):
        d=Images.__new__(Images);d.ids=[0];d.records=[dict(image_id='synthetic',array_index=0,class_index=0)];d.seed=11;d.epoch=0
        d.arr=np.zeros((1,112,112,3),np.uint8);d.arr[:,:,:,0]=255
        d.upstream=GestaltMatcherDataset.__new__(GestaltMatcherDataset);d.upstream.augment=False;d.upstream.target_size=112;d.upstream.aspect_ratio=False;d.upstream.in_channels=3
        x,_,_=d[0];self.assertEqual(float(x[0].mean()),1.);self.assertEqual(float(x[2].mean()),-1.)
        d.upstream.augment=True;torch.manual_seed(1);first=d[0][0];torch.manual_seed(172);second=d[0][0];torch.testing.assert_close(first,second)
    def test_paired_statistics(self):
        from score import interval
        a=np.zeros((4,20,6));b=np.ones_like(a);labels=['a','a','b','c','c','c']
        r=interval(a,b,labels,repetitions=100);self.assertAlmostEqual(r['delta'],1.)
        np.testing.assert_allclose(r['hierarchical_paired_95_CI'],[1.,1.])
    def test_support_patient_separation(self):
        from score import episodes
        x=np.array([[1.,0.]]*10+[[0.,1.]]*10,dtype=np.float32);q=np.array([[1.,0.],[0.,1.],[-1.,0.]],dtype=np.float32)
        out=episodes(x,q,['a']*10+['b']*10,['a','b','unsupported'],5,True)
        self.assertEqual(out[0]['patients'],2);self.assertEqual(out[0]['top1'],1.)
    def test_encoder_freezing_and_original_equivalence(self):
        from onnx2torch import convert
        torch.set_num_threads(4);m=Encoder('r50','wce_feature_only',3).eval();ref=convert(str(WORK/'r50_initial.onnx')).eval();x=torch.randn(2,3,112,112)
        with torch.no_grad():a=m.embedding(x);b=ref(x)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        m.train();self.assertFalse(m.body.training);self.assertTrue(m.feature.training)
        self.assertTrue(all(not p.requires_grad for p in m.body.parameters()));self.assertTrue(all(p.requires_grad for p in m.feature.parameters()))
        del m,ref
        m=Encoder('r50','arcface_full',3).train()
        self.assertTrue(all(not b.training for b in m.modules() if isinstance(b,torch.nn.modules.batchnorm._BatchNorm)))
        slopes=[p for n,p in m.body.named_parameters() if 'onnx_initializer' in n]
        self.assertTrue(slopes);self.assertTrue(all(p.requires_grad for p in slopes));del m
        m=Encoder('transface_chop_cpt','wce_feature_only',3).train()
        self.assertFalse(m.body.training);self.assertTrue(all(not p.requires_grad for p in m.body.parameters()));self.assertTrue(all(p.requires_grad for p in m.feature.parameters()))

if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    save(WORK/'tests.json',dict(state='passed' if result.wasSuccessful() else 'failed',synthetic_only=True,tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),updated_at_utc=now()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
