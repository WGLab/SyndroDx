"""Synthetic mathematical and split checks, without project data."""
import unittest
import numpy as np
from frozen_probe import ridge_predictions, targets, training_folds, select_alpha


class ProbeTests(unittest.TestCase):
    def test_dual_equals_weighted_primal_with_intercept(self):
        rng = np.random.default_rng(2)
        x, q = rng.normal(size=(17, 7)), rng.normal(size=(8, 7))
        terms = [{'a'}] * 9 + [{'b'}] * 7 + [{'a', 'b'}]
        y = targets(terms, ['a', 'b'])
        weights = (y / y.sum(0)).sum(1) / y.sum(1)
        weights *= len(weights) / weights.sum()
        mean, scale = x.mean(0), np.maximum(x.std(0), .01)
        x, q = (x - mean) / scale, (q - mean) / scale
        xm, ym = np.average(x, axis=0, weights=weights), np.average(y, axis=0, weights=weights)
        a, b = (x - xm) * np.sqrt(weights[:, None]), (y - ym) * np.sqrt(weights[:, None])
        coefficient = np.linalg.solve(a.T @ a + 10 * np.eye(7), a.T @ b)
        expected = (q - xm) @ coefficient + ym
        original_x, original_q = x * scale + mean, q * scale + mean
        actual = ridge_predictions(original_x, original_q, terms, ['a', 'b'], [10.])[10.]
        np.testing.assert_allclose(expected, actual, rtol=1e-9, atol=1e-9)

    def test_selection_only_receives_training_data(self):
        terms = [{'a'}] * 15 + [{'b'}] * 15
        x = np.zeros((30, 3))
        x[:15, 0], x[15:, 1] = 1, 1
        fold = training_folds(terms)
        alpha, report = select_alpha(x, terms, fold)
        self.assertEqual(report['training_cv_macro_top1'], 1.)
        self.assertIn(alpha, (.01, .1, 1., 10., 100., 1000., 10000., 100000., 1000000.))
        self.assertEqual(set(fold), set(range(5)))
        np.testing.assert_array_equal(fold, training_folds(terms))


if __name__ == '__main__':
    unittest.main()
