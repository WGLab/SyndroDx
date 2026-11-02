import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evaluation_core import normalized, aggregate_patients, retrieval, macro_score, paired_interval, ridge_probe
import prepare_evaluation as preparation


class EvaluationTests(unittest.TestCase):
    def test_patient_mean_balances_views_and_excludes_other_splits(self):
        records = [dict(group_id='p1', split='test', view='frontal')] * 4 + [dict(group_id='p1', split='test', view='profile'), dict(group_id='p2', split='train', view='frontal')]
        vectors = np.array([[1, 0]] * 4 + [[0, 1], [-1, 0]], np.float32)
        ids, values = aggregate_patients(records, vectors, 'test')
        self.assertEqual(ids, ['p1'])
        np.testing.assert_allclose(values, [[2**-.5, 2**-.5]], atol=1e-6)

    def test_same_patient_is_never_retrieved_as_diagnosis_evidence(self):
        query = np.array([[1, 0]], np.float32)
        gallery = np.array([[1, 0], [.9, .1], [0, 1]], np.float32)
        hits, covered = retrieval(query, gallery, ['p'], ['p', 'q', 'r'], [{'A'}], [{'A'}, {'B'}, {'A'}])
        self.assertFalse(hits[1][0])
        self.assertTrue(hits[5][0])

    def test_macro_is_not_dominated_by_common_class(self):
        self.assertEqual(macro_score([1, 1, 0], [{'common'}, {'common'}, {'rare'}]), .5)
        result = paired_interval(np.zeros(10), np.ones(10), [{'A'}] * 10, repetitions=100)
        self.assertEqual(result['paired_patient_bootstrap_95_interval'], [1, 1])

    def test_nonfinite_or_zero_features_fail(self):
        with self.assertRaises(ValueError):
            normalized([[0, 0]])
        with self.assertRaises(ValueError):
            normalized([[np.nan, 1]])

    def test_frozen_probe_classifies_synthetic_heldout_patients(self):
        rng = np.random.default_rng(2)
        x = np.vstack([rng.normal([-2, 0], .1, (20, 2)), rng.normal([2, 0], .1, (20, 2))]).astype(np.float32)
        y = ['A'] * 20 + ['B'] * 20
        result = ridge_probe(x, x[[1, 21]], x[[2, 22]], y, ['A', 'B'], ['A', 'B'])
        self.assertEqual(result['test_macro'], 1)
        self.assertEqual(result['state'], 'complete')

    def test_private_mapping_exact_user_schema_stays_in_local_join(self):
        import hashlib, hmac
        salt = b'synthetic-test-only'
        key = hmac.new(salt, b'patient:fake-pseudo', hashlib.sha256).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'mapping.csv'
            path.write_text('mrn,psedudo_mrn\n00012,fake-pseudo\n')
            with patch.object(preparation, 'PRIVATE_MAPPING', path):
                bridge = preparation.private_bridge({'patients': {key: {'group_id': 'synthetic-group'}}}, {'synthetic-group'}, salt)
            self.assertEqual(bridge, {'12': {'synthetic-group'}})

    def test_mapping_of_protected_patient_is_excluded(self):
        import hashlib, hmac
        salt = b'synthetic-test-only'
        key = hmac.new(salt, b'patient:fake-pseudo', hashlib.sha256).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'mapping.csv'
            path.write_text('mrn,psedudo_mrn\n00012,fake-pseudo\n')
            with patch.object(preparation, 'PRIVATE_MAPPING', path):
                bridge = preparation.private_bridge({'patients': {key: {'group_id': 'protected-group'}}}, {'released-group'}, salt)
            self.assertFalse(bridge)


if __name__ == '__main__':
    unittest.main()
