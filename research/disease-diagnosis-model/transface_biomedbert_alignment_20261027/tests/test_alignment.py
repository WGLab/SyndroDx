"""Synthetic checks for semantic false negatives, HPO parsing and patient isolation."""
import unittest
import torch
from model import targets,semantic_loss
from prepare import hp_ids,descendants
from common import WORK,load

class Tests(unittest.TestCase):
    def test_same_phenotype_is_positive_across_patients(self):
        h=torch.tensor([[1.,0,0],[1,0,0],[0,1,1]])
        y=torch.tensor([0,1,2]);p,n=targets(h,y)
        self.assertGreater(float(p[0,1]),0)
        self.assertFalse(bool(n[0,1]));self.assertEqual(float(p[0,2]),0)
        torch.testing.assert_close(p.sum(1),torch.ones(3))
    def test_partial_shared_semantics_and_same_disease_are_not_false_negatives(self):
        h=torch.eye(4);h[1]=torch.tensor([1.,1,1,1])
        p,n=targets(h,torch.tensor([0,1,2,2]))
        self.assertTrue(bool(n[0,1]));self.assertTrue(bool(n[2,3]))
    def test_bidirectional_gradients_finite(self):
        image=torch.randn(8,5,requires_grad=True);text=torch.randn(8,5,requires_grad=True)
        loss=semantic_loss(torch.nn.functional.normalize(image,dim=-1),torch.nn.functional.normalize(text,dim=-1),torch.tensor(10.),torch.eye(8),torch.arange(8))
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(image.grad).all());self.assertTrue(torch.isfinite(text.grad).all())
    def test_empty_annotation_not_negative(self):
        self.assertEqual(hp_ids('',{},{}),set())
    def test_released_patient_partitions(self):
        patients=load(WORK/'private/patients.json')['patients']
        a={p['subject'] for p in patients if p['split']=='train'};b={p['subject'] for p in patients if p['split']=='validation'}
        self.assertFalse(a&b)
        self.assertTrue(all(p['hpo'] for p in patients))
        self.assertTrue(all(p['split'] in ('train','validation') for p in patients))
        receipt=load(WORK/'data_receipt.json')
        self.assertTrue(receipt['GMDB_CHOP_BCH_tagged_patients_excluded']);self.assertFalse(receipt['disease_profile_HPO_used_as_patient_observation'])

if __name__=='__main__':unittest.main()
