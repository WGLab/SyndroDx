import unittest
import numpy as np
from evaluate import classify, mapped_scores, retrieve_scores


class EvaluationTests(unittest.TestCase):
    def test_no_input_is_abstention_even_when_zero_score_tie_matches_label(self):
        h=classify(np.zeros((2,3)),np.array([0,-1]),np.array([False,True]))
        self.assertFalse(any(x.any() for x in h.values()))

    def test_exact_equivalent_classes_sum_without_dropping_distractors(self):
        x=mapped_scores(np.array([[.1,.2,.7]]),[0,0,1],2)
        np.testing.assert_allclose(x,[[.3,.7]])

    def test_retrieval_excludes_self(self):
        h=retrieve_scores(np.eye(3),['A','B','C'],np.array([0,1,2]),np.ones(3,bool))
        self.assertFalse(any(v.any() for v in h.values()))


if __name__=='__main__':unittest.main()
