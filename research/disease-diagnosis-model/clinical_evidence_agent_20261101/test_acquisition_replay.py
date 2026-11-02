import unittest
import numpy as np
from acquisition_replay import observations, initial_observations, answer, select_question, pack


class ReplayTests(unittest.TestCase):
    def test_unknown_never_enters_model_input_and_costs_question(self):
        known, asked = {1: 0}, set()
        self.assertEqual(answer({1: 0}, known, asked, 2), 'unknown')
        self.assertEqual(known, {1: 0})
        self.assertEqual(asked, {2})
        with self.assertRaises(ValueError):
            answer({1: 0}, known, asked, 2)

    def test_explicit_negative_is_preserved(self):
        known, asked = {}, set()
        self.assertEqual(answer({2: 1}, known, asked, 2), 'absent')
        self.assertEqual(known, {2: 1})

    def test_conflicts_and_unresolved_are_not_answers(self):
        self.assertEqual(observations([1, 1, 2, 3], [0, 1, 2, 0]), {3: 0})

    def test_paired_nested_initial_masks(self):
        r = {i: 0 for i in range(1, 9)}; r[10] = 1
        a = initial_observations(r, 1, 11, 0)
        b = initial_observations(r, 3, 11, 0)
        self.assertLessEqual(set(a), set(b))
        self.assertEqual(a[10], 1)
        self.assertEqual(a, initial_observations(r, 1, 11, 0))

    def test_policy_does_not_see_hidden_answers(self):
        inverse = {1: 'HP:0000001', 2: 'HP:0000002'}
        args = ('adaptive', [.9, .1], ['A', 'B'], set(), [1, 2], [2, 1],
                {'A': ['HP:0000001'], 'B': ['HP:0000002']}, 'a'*64,
                {v: k for k, v in inverse.items()}, inverse)
        first = select_question(*args, np.random.default_rng(1))
        known, asked = {}, set()
        self.assertEqual(answer({}, known, asked, first), 'unknown')
        second = select_question(*args, np.random.default_rng(1))
        self.assertEqual(first, second)

    def test_pack_preserves_assertions(self):
        ids, states = pack([{2: 1, 1: 0}], {1: 'HP:0000001', 2: 'HP:0000002'})
        self.assertEqual(ids[0, :2].tolist(), [1, 2])
        self.assertEqual(states[0, :2].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
