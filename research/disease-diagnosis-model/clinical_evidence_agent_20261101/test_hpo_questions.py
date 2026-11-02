import unittest
from hpo_questions import select
from engine import run
from test_engine import case, observation


class AssociationQuestionTests(unittest.TestCase):
    def test_uses_only_explicit_associations_and_excludes_known_terms(self):
        profiles = {'A': ['HP:0000001'], 'B': ['HP:0000002']}
        q = select({'A': 2., 'B': 0.}, {'HP:0000001'}, profiles, 'a' * 64)
        self.assertEqual(q['concept_id'], 'HP:0000002')
        self.assertFalse(q['unlisted_association_means_absent'])
        self.assertNotIn('planning_information_gain', q)

    def test_no_profiles_produce_no_question(self):
        self.assertIsNone(select({'A': 0.}, set(), {}, 'a' * 64))

    def test_public_knowledge_cannot_fill_patient_answer(self):
        choose = lambda scores, excluded: select(scores, excluded, {'A': ['HP:0000001']}, 'a' * 64)
        r = run(case(), lambda _: {'A': 0.}, question_selector=choose)
        self.assertEqual(len(r['trace']), 1)
        self.assertEqual(r['trace'][0]['outcome'], 'unavailable_unknown')
        self.assertFalse(r['accepted_observation_ids'])

    def test_selector_cannot_request_arbitrary_tools(self):
        with self.assertRaises(ValueError):
            run(case(), lambda _: {'A': 0.}, question_selector=lambda *_: {'tool': 'external_api'})


if __name__ == '__main__':
    unittest.main()
