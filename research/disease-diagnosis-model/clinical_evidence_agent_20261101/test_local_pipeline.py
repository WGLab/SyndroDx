import unittest
from local_pipeline import LocalPipeline
from test_engine import case, observation


class FakeAdapter:
    manifest = {'artifacts': {'visual_encoder': {'sha256': 'a' * 64}}}

    def bind(self, feature, mode):
        return lambda evidence: {'SYNTHETIC': 0.}


class FakeVariantIndex:
    def cards(self, patient_key, evidence, cutoff):
        return [{'status': 'synthetic_candidate_only', 'used_for_diagnostic_ranking': False} for _ in evidence]


class LocalPipelineTests(unittest.TestCase):
    def test_optional_variant_does_not_block_hpo_question(self):
        choose = lambda scores, excluded: None if 'HP:0000002' in excluded else {'tool': 'request_hpo_observation', 'concept_id': 'HP:0000002'}
        pipeline = LocalPipeline(FakeAdapter(), choose, FakeVariantIndex())
        result = pipeline.assess(case([observation(kind='variant', assertion='unknown')]))
        self.assertEqual(result['trace'][0]['action']['tool'], 'request_hpo_observation')
        self.assertEqual(result['optional_variant_records_held_for_review'], 1)
        self.assertFalse(result['variants_used_for_ranking'])
        self.assertEqual(len(result['variant_evidence_cards']), 1)

    def test_variant_review_does_not_change_diagnosis(self):
        pipeline = LocalPipeline(FakeAdapter(), lambda *_: None, FakeVariantIndex())
        with_variant = pipeline.assess(case([observation(kind='variant', assertion='unknown')]))
        without_variant = pipeline.assess(case())
        self.assertEqual(with_variant['differential'], without_variant['differential'])

    def test_unverified_visual_feature_source_rejected(self):
        pipeline = LocalPipeline(FakeAdapter(), lambda *_: None)
        with self.assertRaises(ValueError):
            pipeline.assess(case(), native512_feature=[0] * 512)

    def test_optional_variant_still_requires_correct_patient(self):
        pipeline = LocalPipeline(FakeAdapter(), lambda *_: None)
        with self.assertRaises(ValueError):
            pipeline.assess(case([observation(kind='variant', patient_key='another_synthetic_patient')]))


if __name__ == '__main__':
    unittest.main()
