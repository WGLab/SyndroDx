import unittest
from copy import deepcopy
from variant_review import review_card
from test_engine import observation


def record():
    obs = observation(kind='variant', assertion='unknown', review_status='pending',
                      identity_verified=False, assay_scope_verified=False)
    candidate = {'patient_key': obs['patient_key'], 'variant_key': obs['evidence_id'], 'source_gene': 'SYNTHETIC',
                 'matches': [{'VariationID': '1', 'match_type': 'exact_versioned_nucleotide_expression'}],
                 'public_variant_assertions': {'1': {'classification': 'Pathogenic'}}}
    return {'patient_key': obs['patient_key'], 'variant_key': obs['evidence_id'],
            'observation': obs, 'candidate_knowledge': candidate}


class VariantCardTests(unittest.TestCase):
    def test_public_pathogenic_assertion_does_not_confirm_patient(self):
        r = record(); c = review_card(r, r['patient_key'], None, {})
        self.assertEqual(c['patient_detection_status'], 'unresolved')
        self.assertFalse(c['local_causality_asserted'])
        self.assertFalse(c['used_for_diagnostic_ranking'])
        self.assertEqual(c['rank_adjustment'], 0)
        self.assertIn('missing_decision_time', c['observation_gate_reasons'])

    def test_cross_patient_join_rejected(self):
        r = record(); r['candidate_knowledge']['patient_key'] = 'different_synthetic_patient'
        with self.assertRaises(ValueError): review_card(r, r['patient_key'], None, {})

    def test_ambiguous_identity_not_promoted(self):
        r = record()
        m = deepcopy(r['candidate_knowledge']['matches'][0]); m['VariationID'] = '2'
        r['candidate_knowledge']['matches'].append(m)
        self.assertFalse(review_card(r, r['patient_key'], None, {})['unique_exact_versioned_nucleotide_match'])

    def test_protein_only_not_promoted_to_exact_allele(self):
        r = record(); r['candidate_knowledge']['matches'][0]['match_type'] = 'protein_expression_only_candidate'
        c = review_card(r, r['patient_key'], None, {})
        self.assertEqual(c['clinvar_candidate_ids'], [])


if __name__ == '__main__': unittest.main()
