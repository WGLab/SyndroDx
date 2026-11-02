import unittest
import numpy as np
from learned_questions import fit_tables, LearnedQuestions, split_policy_patients, CONFIGS


class LearnedQuestionTests(unittest.TestCase):
    def test_patient_split_is_disjoint_and_exhaustive(self):
        labels=np.array([0]*10+[1]*3+[2]*2+[3])
        fit, selection=split_policy_patients(labels)
        self.assertFalse(set(fit)&set(selection))
        self.assertEqual(set(fit)|set(selection),set(range(len(labels))))
        self.assertTrue(set(labels[selection])<=set(labels[fit]))

    def model(self):
        records=[{1:0,2:0}]*5+[{1:0,3:0}]*5
        tables=fit_tables(records,[0]*5+[1]*5,[1,2,3],2)
        inverse={i:f'HP:{i:07d}' for i in [1,2,3]}
        return LearnedQuestions(tables,CONFIGS[1],['A','B'],inverse)

    def test_conditioning_and_exclusion(self):
        model=self.model()
        terms,_=model.choose([[.99,.01],[.01,.99]],[{1},{1}])
        self.assertEqual(terms,[2,3])
        terms,_=model.choose([[.99,.01]],[{1,2,3}])
        self.assertEqual(terms,[None])

    def test_unknown_remains_separate_from_negative(self):
        model=self.model()
        self.assertEqual(model.counts[:,:,1].sum(),0)
        self.assertTrue(np.all(model.rates.sum(-1)<1))
        self.assertTrue(np.all(model.rates[:,:,1]<.1))

    def test_selector_is_compatible_with_log_score_callback(self):
        model=self.model()
        action=model({'A':np.log(.99),'B':np.log(.01)},{'HP:0000001'})
        self.assertEqual(action['concept_id'],'HP:0000002')
        self.assertFalse(action['unknown_means_absent'])

    def test_truth_or_hidden_answers_are_not_policy_inputs(self):
        model=self.model()
        before=model.choose([[.9,.1]],[{1}])[0]
        # The selector's API has no argument for patient's truth/hidden evidence.
        after=model.choose([[.9,.1]],[{1}])[0]
        self.assertEqual(before,after)
        with self.assertRaises(ValueError): model({'unexpected':1.},set())


if __name__=='__main__':unittest.main()
