import unittest
from audit import canonical, exact_class_mapping


class CrosswalkTests(unittest.TestCase):
    def test_multiple_omim_needs_same_exact_target(self):
        exact={'OMIM:100001':{'MONDO:0000001'},'OMIM:100002':{'MONDO:0000001'},'OMIM:100003':{'MONDO:0000002'}}
        self.assertEqual(exact_class_mapping('100001;100002',exact),'MONDO:0000001')
        self.assertIsNone(exact_class_mapping('100001;100003',exact))
        self.assertIsNone(exact_class_mapping('100001;100004',exact))

    def test_ambiguous_exact_mapping_not_guessed(self):
        self.assertIsNone(exact_class_mapping('100001',{'OMIM:100001':{'MONDO:0000001','MONDO:0000002'}}))

    def test_alias_and_unique_replacement(self):
        terms={'A':{'obsolete':False,'replaced_by':[]},'OLD':{'obsolete':True,'replaced_by':['A']}}
        self.assertEqual(canonical('ALT',terms,{'ALT':'OLD'}),'A')
        self.assertIsNone(canonical('MISSING',terms,{}))

    def test_ambiguous_obsolete_and_cycle_not_guessed(self):
        self.assertIsNone(canonical('OLD',{'OLD':{'obsolete':True,'replaced_by':['A','B']}},{}))
        self.assertIsNone(canonical('A',{}, {'A':'B','B':'A'}))


if __name__=='__main__':unittest.main()
