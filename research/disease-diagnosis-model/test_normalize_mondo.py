import unittest

from normalize_mondo import exact_index, normalize_raw
from ontology import Term


class NormalizationSafetyTests(unittest.TestCase):
    def test_broad_and_narrow_synonyms_do_not_become_exact(self):
        t = Term('MONDO:1')
        t.name = 'specific disease'
        t.synonyms = [('general disease', 'BROAD'), ('subtype', 'NARROW'),
                      ('same disease', 'EXACT')]
        idx = exact_index({t.id: t}, 'MONDO:')
        self.assertNotIn('general disease', idx)
        self.assertNotIn('subtype', idx)
        self.assertEqual(idx['same disease'], {t.id})

    def test_ambiguous_exact_matches_are_not_arbitrarily_selected(self):
        r = normalize_raw('same name', {'same name': {'MONDO:1', 'MONDO:2'}}, {}, {})
        self.assertEqual(r['status'], 'needs_review')
        self.assertEqual(r['mondo_id'], '')

    def test_nf_does_not_become_nf1(self):
        r = normalize_raw('NF', {}, {}, {'nf': 'MONDO:0021061'})
        self.assertEqual(r['mondo_id'], 'MONDO:0021061')

    def test_unqualified_nevus_does_not_become_large_congenital_nevus(self):
        r = normalize_raw('congenital melanocytic nevus',
                          {'congenital melanocytic nevus': {'MONDO:0044792'}}, {}, {})
        self.assertEqual(r['status'], 'needs_review')
        self.assertEqual(r['mondo_id'], '')

    def test_combined_syndromes_do_not_inherit_single_alias(self):
        r = normalize_raw('aperts pfeiffer', {}, {}, {'aperts': 'MONDO:0007041'})
        self.assertEqual(r['status'], 'unresolved')
        self.assertEqual(r['mondo_id'], '')

    def test_laterality_loss_is_explicit(self):
        r = normalize_raw('UCLP', {}, {}, {'uclp': 'MONDO:0016044'})
        self.assertEqual(r['mapping_relation'], 'skos:broadMatch')

    def test_phenotype_does_not_force_disease(self):
        r = normalize_raw('macrocephaly', {}, {'macrocephaly': {'HP:0000256'}}, {})
        self.assertEqual(r['status'], 'hpo_only')
        self.assertEqual(r['mondo_id'], '')

    def test_synostosis_retains_subtype_but_ridge_is_not_diagnosed(self):
        r = normalize_raw('metopic syn', {}, {}, {})
        self.assertEqual(r['mondo_id'], 'MONDO:0015469')
        self.assertEqual(r['hpo_ids'], 'HP:0011330')
        self.assertEqual(r['mapping_relation'], 'skos:broadMatch')
        self.assertEqual(normalize_raw('metopic ridge', {}, {}, {})['mondo_id'], '')

    def test_port_wine_stain_does_not_imply_familial_disease(self):
        r = normalize_raw('port wine stain', {'port wine stain': {'MONDO:0008094'}}, {}, {})
        self.assertEqual(r['status'], 'needs_review')
        self.assertEqual(r['mondo_id'], '')


if __name__ == '__main__':
    unittest.main()
