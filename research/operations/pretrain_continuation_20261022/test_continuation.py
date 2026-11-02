"""Synthetic-only tests. No datasets, checkpoints, GPUs, or private logs read."""
import ast
import math
from pathlib import Path
import tempfile
import unittest

import continue_pretrain as continuation


def synthetic_metrics(slope=0, oscillation=0):
    result = []
    start = continuation.PLAN['start_iteration']
    for offset in range(0, 1000, 10):
        value = 14 + slope * offset + oscillation * math.sin(offset)
        result.append({'iteration': start + offset, 'total_loss': value,
                       'dino_global_crops_loss': 4.0, 'dino_local_crops_loss': 5.0,
                       'ibot_loss': 5.1, 'koleo_loss': -0.1})
    return result


class ContinuationTests(unittest.TestCase):
    def test_early_plateau_cannot_stop_training(self):
        self.assertEqual(continuation.plateau(synthetic_metrics(), 500)['state'], 'continue')

    def test_stable_training_loss_can_stop_after_minimum(self):
        self.assertEqual(continuation.plateau(synthetic_metrics(oscillation=0.001), 1000)['state'],
                         'training_loss_plateau')

    def test_meaningfully_improving_loss_continues(self):
        self.assertEqual(continuation.plateau(synthetic_metrics(slope=-0.001), 1000)['state'], 'continue')

    def test_diverging_loss_does_not_look_like_plateau(self):
        self.assertEqual(continuation.plateau(synthetic_metrics(slope=0.001), 1000)['state'], 'continue')

    def test_nonfinite_loss_requires_review(self):
        rows = synthetic_metrics()
        rows[-1]['total_loss'] = float('nan')
        self.assertEqual(continuation.plateau(rows, 1000)['state'], 'review_required')

    def test_component_drift_cannot_hide_behind_total_loss(self):
        rows = synthetic_metrics()
        for index, row in enumerate(rows):
            row['ibot_loss'] += index * 0.02
        self.assertEqual(continuation.plateau(rows, 1000)['state'], 'continue')

    def test_missing_components_requires_review(self):
        rows = synthetic_metrics()
        for row in rows:
            del row['ibot_loss']
        self.assertEqual(continuation.plateau(rows, 1000)['state'], 'review_required')

    def test_sparse_metrics_requires_review(self):
        self.assertEqual(continuation.plateau(synthetic_metrics()[::10], 1000)['state'], 'review_required')

    def test_numeric_parser_never_returns_other_fields(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'synthetic.jsonl'
            path.write_text('{"iteration":2136,"total_loss":14.0,"unrelated":"SYNTHETIC"}\n')
            self.assertEqual(continuation.read_numeric_metrics(path), [{'iteration': 2136, 'total_loss': 14.0}])


if __name__ == '__main__':
    unittest.main()
