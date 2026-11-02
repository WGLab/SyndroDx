import unittest
from run_agent import exact_resolver


class ResolverTests(unittest.TestCase):
    def test_missing_record_is_unknown(self):
        resolver=exact_resolver('synthetic_patient',[])
        self.assertIsNone(resolver({'tool':'request_hpo_observation','concept_id':'HP:0000001'}))

    def test_cross_patient_answers_rejected(self):
        with self.assertRaises(ValueError):
            exact_resolver('synthetic_patient',[{'patient_key':'other_synthetic'}])

    def test_duplicate_answer_requires_review(self):
        r={'patient_key':'synthetic_patient','kind':'hpo','concept_id':'HP:0000001'}
        resolver=exact_resolver('synthetic_patient',[r,r])
        self.assertIsNone(resolver({'tool':'request_hpo_observation','concept_id':'HP:0000001'}))


if __name__=='__main__':unittest.main()
