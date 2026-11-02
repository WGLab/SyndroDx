"""Patient holdout must survive ID formatting, aliases, paths and exact copies."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from expert_reservation import filter_reserved
from audit_expert_holdout import mids

class ReservationTest(unittest.TestCase):
    def test_aliases_paths_hashes_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'protected_mrns.tsv').write_text('mrn\n123\n')
            (p/'blocked_paths.tsv').write_text('path\n/reserved/face.jpg\n')
            (p/'blocked_crop_hashes.tsv').write_text('sha256\nabc\n')
            rows=[{'mrn':'00000123'}, {'mrn':'123.0'},
                  {'mrn':'9','identity_candidates':'8|123'},
                  {'mrn':'9','cropping_path':'/reserved//face.jpg'},
                  {'mrn':'9','crop_sha256':'abc'},
                  {'mrn':'9','cropping_path':'/other.jpg'}]
            kept,n=filter_reserved(rows,p)
            self.assertEqual(n,5);self.assertEqual(kept,rows[-1:])
            with self.assertRaises(FileNotFoundError):filter_reserved([],p/'absent')

    def test_mondo_ids_are_not_silently_truncated(self):
        self.assertEqual(mids("['MONDO:00081047']"),set())
        self.assertEqual(mids("['MONDO:0032690  ']"),{'MONDO:0032690'})

    def test_real_legacy_manifest_is_filtered(self):
        with open('labels.tsv') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        kept,n=filter_reserved(rows)
        s=json.loads(Path('expert_holdout_audit/summary.json').read_text())
        self.assertGreaterEqual(n,s['historical_label_rows_reserved'])
        with open('split_plan_v0/ian_protected_mrns.tsv') as f:
            ian={r['mrn'] for r in csv.DictReader(f,delimiter='\t')}
        from audit_metadata_sources import mrnkey
        self.assertFalse({mrnkey(r['mrn']) for r in kept} & ian)
        self.assertEqual(len(kept)+n,len(rows))
        result={'passed':True,'legacy_rows_excluded':n,'legacy_rows_remaining':len(kept)}
        Path('expert_holdout_audit/guard_validation.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':unittest.main()
