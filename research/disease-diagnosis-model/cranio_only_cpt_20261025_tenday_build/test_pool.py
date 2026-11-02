"""Synthetic allocation and exactly-once diagnostic-claim checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pool_work
from pool_controller import select_resources,resources,SchedulerUnavailable


class ResourceTests(unittest.TestCase):
    def fixture(self):
        registry={'jobs':{str(i):str(100+i) for i in range(20)}}
        records={str(100+i):dict(JobId=str(100+i),UserId='research_user(1)',JobState='RUNNING',
            Partition='gpu-xe9680q',NodeList='node'+str(i//8),AllocTRES='cpu=4,mem=96G,gres/gpu=1') for i in range(20)}
        def receipt(path):
            job=path.stem
            return dict(job_id=job,node=records[job]['NodeList'])
        return registry,records,receipt
    def test_twenty_single_gpu_allocations_form_one_pool(self):
        r,d,f=self.fixture()
        with patch('pool_controller.load',side_effect=f):self.assertEqual(len(select_resources(r,d)),20)
    def test_pending_cards_are_not_counted_as_allocated(self):
        r,d,f=self.fixture();d['100']['JobState']='PENDING'
        with patch('pool_controller.load',side_effect=f):self.assertEqual(len(select_resources(r,d)),19)
    def test_duplicate_identity_owner_and_multiple_gpus_rejected(self):
        r,d,f=self.fixture();r['jobs']['1']='100'
        with self.assertRaises(RuntimeError):select_resources(r,d)
        for key,value in [('UserId','other(1)'),('AllocTRES','cpu=4,mem=96G,gres/gpu=2')]:
            r,d,f=self.fixture();d['100'][key]=value
            with self.assertRaises(RuntimeError):select_resources(r,d)


class SchedulerTests(unittest.TestCase):
    def test_transient_scheduler_failure_is_explicitly_retryable(self):
        import subprocess
        registry={'jobs':{'0':'100'}}
        with patch('pool_controller.load',return_value=registry),patch('pool_controller.query',side_effect=subprocess.TimeoutExpired('squeue',15)):
            with self.assertRaises(SchedulerUnavailable):resources()


class TaskClaimTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'staging').mkdir()
        (self.root/'staging/ledger.json').write_text(json.dumps({'tasks':{str(i):dict(state='pending') for i in range(3)}}))
        self.patch=patch.object(pool_work,'WORK',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_no_task_is_assigned_to_two_workers(self):
        keys=[pool_work.claim(str(i))[0] for i in range(3)]
        self.assertEqual(len(set(keys)),3);self.assertIsNone(pool_work.claim('fourth'))
    def test_completed_tasks_are_not_replayed(self):
        key,_=pool_work.claim('one');pool_work.finish(key,'one',complete=True)
        rest=[pool_work.claim(str(i))[0] for i in range(2)]
        self.assertNotIn(key,rest)
    def test_yielded_task_can_resume_with_a_new_owner(self):
        key,_=pool_work.claim('one');pool_work.finish(key,'one',complete=False)
        again,_=pool_work.claim('two');self.assertEqual(key,again)
    def test_quarantined_chunks_are_not_replayed_or_admitted(self):
        key,_=pool_work.claim('one');pool_work.finish(key,'one',complete=False,quarantined=True)
        self.assertEqual(json.loads((self.root/'staging/ledger.json').read_text())['tasks'][key]['state'],'needs_review')
        rest=[pool_work.claim(str(i))[0] for i in range(2)]
        self.assertNotIn(key,rest)
    def test_wrong_worker_cannot_complete_a_claim(self):
        key,_=pool_work.claim('one')
        with self.assertRaises(RuntimeError):pool_work.finish(key,'two',complete=True)


if __name__=='__main__':unittest.main()
