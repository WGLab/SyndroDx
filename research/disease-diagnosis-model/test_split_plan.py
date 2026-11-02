import collections
import csv
import unittest
from pathlib import Path
from plan_expert_splits import allocate


class SplitPlanTest(unittest.TestCase):
    def test_small_classes_and_determinism(self):
        for n in (1, 4, 5, 11, 20, 60, 736):
            rows=[{'mrn':str(i), 'source_tags':'a' if i%3 else 'b'} for i in range(n)]
            other=[dict(r) for r in reversed(rows)]
            allocate(rows); allocate(other)
            self.assertEqual({r['mrn']:r['proposed_role'] for r in rows},
                             {r['mrn']:r['proposed_role'] for r in other})
            c=collections.Counter(r['proposed_role'] for r in rows)
            self.assertGreater(c['train_candidate'],0)
            if n<5:self.assertEqual(c['train_candidate'],n)
            if n<20:self.assertEqual(c['test_candidate'],0)

    def test_image_and_patient_groups_never_cross_roles(self):
        root=Path('split_plan_v0')
        with (root/'patient_split_proposal.tsv').open() as f:
            patients=list(csv.DictReader(f,delimiter='\t'))
        groups=collections.defaultdict(set)
        for r in patients:
            groups[r['group_id']].add(r['proposed_role'])
            self.assertEqual(r['training_admitted'],'False')
            if r['ian_linked']=='True':self.assertEqual(r['proposed_role'],'expert_test_reserve')
            if r['proposed_role']=='test_candidate':
                self.assertEqual(r['target_coverage_in_proposed_training'],'all')
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        keys=collections.defaultdict(set)
        with (root/'image_split_proposal.tsv').open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                for k in ('image_path','cropping_path','crop_sha256'):
                    if r[k]:keys[(k,r[k])].add(r['proposed_role'])
        self.assertTrue(all(len(v)==1 for v in keys.values()))

if __name__=='__main__':unittest.main()
