import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import crop_handoff_controller as c


class HandoffTests(unittest.TestCase):
    def test_only_completed_idle_replicas_are_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'replicas').mkdir()
            for slot, state in enumerate(('annotating', 'worker_finished', 'worker_finished', 'worker_finished')):
                c.save(run/'replicas'/f'slot_{slot:02d}.json', {'state':state})
            items = [{'job':str(i), 'replica':i} for i in range(4)]
            jobs = {str(i):'RUNNING' for i in range(4)}
            self.assertEqual(c.idle_allocations(run, items, jobs, {'2'}, {'3'}), [items[1]])
            c.save(run/'completion.json', {'state':'annotation_complete'})
            self.assertEqual(c.idle_allocations(run, items, jobs, {'2'}, {'3'}), items[:2])
            jobs['1'] = 'PENDING'
            self.assertEqual(c.idle_allocations(run, items, jobs, {'2'}, {'3'}), items[:1])

    def fixture(self, run, root, missing=False, wrong_patient=False, duplicate=False):
        (root/'workers').mkdir(parents=True)
        (root/'records').mkdir()
        c.save(run/'completion.json', {'state':'annotation_complete'})
        fields = ['photo_id','patient_key','original_path']
        with (run/'accepted_faces.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerow({'photo_id':'synthetic-a','patient_key':'synthetic-p','original_path':'synthetic-source'})
        for i in range(2):
            c.save(root/'workers'/f'slot_{i:02d}.json', {'state':'finished_provisional'})
            db=sqlite3.connect(root/'records'/f'slot_{i:02d}.sqlite')
            db.execute('CREATE TABLE records(record TEXT)')
            if (i == 0 or duplicate) and not missing:
                value={'photo_id':'synthetic-a','patient_key':'other' if wrong_patient else 'synthetic-p',
                       'status':'hold','view':'profile','reason':'synthetic_hold'}
                db.execute('INSERT INTO records VALUES (?)', (json.dumps(value),))
            if i == 1:
                value={'photo_id':'synthetic-excluded','patient_key':'synthetic-p','status':'hold','view':'profile'}
                db.execute('INSERT INTO records VALUES (?)', (json.dumps(value),))
            db.commit();db.close()

    def test_final_reconciliation_excludes_unaccepted_and_keeps_holds(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);root=run/'crop_pipeline'
            self.fixture(run, root)
            self.assertTrue(c.finalize(root, 2))
            completion=c.load(run/'crop_completion.json')
            self.assertFalse(completion['training_admitted'])
            self.assertEqual(completion['counts'], {'hold':1})
            self.assertEqual(completion['provisional_records_excluded_by_final_annotation'], 1)
            with (run/'crop_hold.tsv').open() as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual([row['photo_id'] for row in rows], ['synthetic-a'])

    def test_missing_coverage_identity_conflict_and_duplicate_ownership_block_completion(self):
        for kwargs in ({'missing':True}, {'wrong_patient':True}, {'duplicate':True}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as folder:
                run=Path(folder);root=run/'crop_pipeline'
                self.fixture(run,root,**kwargs)
                with self.assertRaises(RuntimeError):c.finalize(root,2)
                self.assertFalse((run/'crop_completion.json').exists())

    def test_successful_crop_requires_saved_file_and_remains_provisional(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);root=run/'crop_pipeline'
            self.fixture(run,root)
            value={'photo_id':'synthetic-a','patient_key':'synthetic-p','status':'validated_provisional',
                   'view':'profile','output_path':'synthetic.png',
                   'output':{'width':8,'height':8,'sha256':'synthetic-digest'}}
            db=sqlite3.connect(root/'records/slot_00.sqlite')
            db.execute('UPDATE records SET record=?',(json.dumps(value),));db.commit();db.close()
            with self.assertRaises(RuntimeError):c.finalize(root,2)
            self.assertFalse((run/'crop_completion.json').exists())
            (root/'synthetic.png').write_bytes(b'synthetic-file-placeholder')
            self.assertTrue(c.finalize(root,2))
            with (run/'cropped_candidates.tsv').open() as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(rows[0]['training_admitted'],'false')
            self.assertEqual(c.load(run/'crop_completion.json')['counts'],{'validated_provisional':1})

    def test_no_crop_launch_before_annotation_worker_is_finished(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);root=run/'crop_pipeline'
            (root/'workers').mkdir(parents=True);(run/'replicas').mkdir();(run/'direct_workers').mkdir()
            c.save(root/'READY.json', {'worker_count':2})
            c.save(run/'replicas/slot_00.json', {'state':'annotating'})
            items=[{'job':'100','replica':0}]
            with patch.object(c,'allocations',return_value=items),patch.object(c,'scheduler',return_value=({'100':'RUNNING'},set())),patch.object(c,'notify'),patch.object(c.subprocess,'Popen') as launch:
                c.tick(root)
                launch.assert_not_called()
            self.assertEqual(c.load(root/'handoff_status.json')['state'],'waiting_for_annotation_worker_to_finish')

    def test_launch_uses_exclusive_step_in_existing_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);root=run/'crop_pipeline'
            (root/'workers').mkdir(parents=True);(run/'replicas').mkdir();(run/'direct_workers').mkdir()
            c.save(root/'READY.json', {'worker_count':2})
            c.save(run/'replicas/slot_00.json', {'state':'worker_finished'})
            with patch.object(c,'allocations',return_value=[{'job':'100','replica':0}]),patch.object(c,'scheduler',return_value=({'100':'RUNNING'},set())),patch.object(c,'notify'),patch.object(c.subprocess,'Popen') as launch:
                launch.return_value.pid=123
                c.tick(root)
                args=launch.call_args.args[0]
                self.assertIn('--jobid=100',args)
                self.assertIn('--exclusive',args)
                self.assertNotIn('--overlap',args)
                self.assertEqual(args[-3:], ['/bin/bash',str(root/'launch_crop.sh'),'0'])
                c.tick(root)
                self.assertEqual(launch.call_count,1)


if __name__ == '__main__':unittest.main(verbosity=2)
