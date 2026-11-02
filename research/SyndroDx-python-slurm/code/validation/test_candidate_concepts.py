import unittest
from normalize_candidate_concepts import Normalizer,key
class ConceptsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.n=Normalizer()
 def one(self,x):return self.n.map(x)[0][0]
 def test_no_etiology_from_shape(self):
  for x in ['hemihypertrophy','scaphocephaly','trigonocephaly','plagiocephaly','metopic ridge','pierre robin']:
   self.assertEqual(self.one(x)['mondo_ids'],'',x)
  self.assertEqual(self.one('sagittal syn')['mondo_ids'],'MONDO:0015469')
 def test_abbreviations_and_acronym_collision(self):
  for x in ['CP','CL','22q 11.2','NF','HFM']:
   self.assertEqual(self.one(x)['mondo_ids'],'',x)
  self.assertEqual(self.one('scar')['hpo_ids'],'HP:0100699')
  self.assertEqual(self.one('scar')['mondo_ids'],'')
 def test_do_not_delete_real_conditions_as_procedures(self):
  for x in ['gynecomastia','macromastia','gender dysphoria']:
   self.assertNotEqual(self.one(x)['concept_type'],'procedure_or_workflow')
  self.assertEqual(self.one('26.3 rhino')['concept_type'],'procedure_or_workflow')
  self.assertEqual(key('22q 11.2 del'),'22q 11 2 del')
 def test_mixed_labels_are_not_single_diagnoses(self):
  for x in ['UCLP/CP/UCLCP','aperts pfeiffer','rhinoplasty; micrognathia','CP-PR']:
   rr,relation=self.n.map(x)
   self.assertNotEqual(relation,'single',x)
   self.assertGreater(len(rr),1)
  self.assertEqual(self.n.map('cleft lip/palate')[1],'single')
 def test_suspicion_not_positive(self):
  self.assertEqual(self.one('suspected Noonan syndrome')['mondo_ids'],'')
 def test_gene_not_syndrome_and_cleft_is_structural(self):
  self.assertEqual(self.one('FGFR3')['concept_type'],'gene_only')
  self.assertEqual(self.one('UCLP')['concept_type'],'structural_diagnosis')
  self.assertEqual(self.one('UCLP')['mapping_relation'],'broader_concept')
if __name__=='__main__':unittest.main()
