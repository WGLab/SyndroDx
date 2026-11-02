import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import macro_hits,validation_metrics
from prepare import label_budget

class ClinicalMetrics(unittest.TestCase):
    def test_macro_does_not_overweight_common_class(self):
        scores=np.array([[2.,0.]]*9+[[2.,0.]])
        self.assertEqual(macro_hits(scores,[0]*9+[1])['top1'],.5)
    def test_patient_averaging_differs_from_image_vote(self):
        rows=[dict(subject='synthetic_a',class_index=0),dict(subject='synthetic_a',class_index=0),dict(subject='synthetic_b',class_index=1)]
        values=np.array([[4.,0.],[0.,3.],[0.,4.]])
        actual=validation_metrics(values,[0,1,2],rows)
        self.assertEqual(actual['patients'],2);self.assertEqual(actual['patient_macro']['top1'],1.)
        self.assertEqual(actual['image_macro']['top1'],.75)
    def test_budget_keeps_all_images_for_chosen_patients_and_classes(self):
        rows=[dict(subject=f'synthetic_{i}',label=str(i%2),array_index=2*i+j) for i in range(10) for j in range(2)]
        indices,n=label_budget(rows,.25,1)
        self.assertEqual(len(indices),n*2)
        self.assertEqual({rows[i]['label'] for i in indices},{'0','1'})
        self.assertEqual(label_budget(rows,.25,1),(indices,n))

if __name__=='__main__':unittest.main()
