import unittest
from copy import deepcopy
from compare import evaluate


def records():
    output = []
    for arm in ['fixed', 'agent']:
        for i in range(4):
            output.append(dict(arm=arm, patient_key=f'synthetic_p{i}', group_id=f'synthetic_g{i // 2}',
                split_audit_passed=True, truth_adjudicated=True, truth_source_sha256='a' * 64,
                disposition='validation', budget=5, actions_used=0 if arm == 'fixed' else 1,
                decision_time='2026-01-02T00:00:00+00:00', candidate_universe=['A', 'B'],
                ranking=['A', 'B'] if arm == 'fixed' else ['B', 'A'], truth_ids=['B'],
                initial_information_sha256='b' * 64, available_information_sha256='c' * 64))
    return output


class ComparisonTests(unittest.TestCase):
    def evaluate(self, data, groups=None):
        return evaluate(data, groups or set(), ['fixed', 'agent'], 5, bootstrap=40)

    def test_paired_scores_and_cluster_bootstrap(self):
        r = self.evaluate(records())
        self.assertEqual(r['patients'], 4)
        self.assertEqual(r['patient_groups'], 2)
        self.assertEqual(r['differences_from_reference']['agent']['top1']['delta_patient_micro'], 1)
        self.assertEqual(r['differences_from_reference']['agent']['top1']['paired_patient_group_bootstrap_ci95'], [1, 1])

    def test_patient_group_leakage_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(records(), {'synthetic_g0'})

    def test_unmatched_patient_or_candidate_set_rejected(self):
        for altered in [records()[:-1], records()]:
            if len(altered) == 8:
                altered[-1]['candidate_universe'].append('C')
            with self.assertRaises(ValueError):
                self.evaluate(altered)

    def test_unadjudicated_truth_rejected(self):
        r = records()
        r[0]['truth_adjudicated'] = False
        with self.assertRaises(ValueError):
            self.evaluate(r)

    def test_out_of_universe_truth_kept_as_miss(self):
        r = records()
        for row in r:
            if row['patient_key'] == 'synthetic_p0':
                row['truth_ids'] = ['UNSEEN']
        result = self.evaluate(r)
        self.assertEqual(result['truth_outside_candidate_universe_patients'], 1)
        self.assertEqual(result['arms']['agent']['top5_patient_micro'], .75)

    def test_unequal_budget_or_available_information_rejected(self):
        for key, value in [('budget', 6), ('available_information_sha256', 'd' * 64)]:
            r = records()
            r[-1][key] = value
            with self.assertRaises(ValueError):
                self.evaluate(r)


if __name__ == '__main__':
    unittest.main()
