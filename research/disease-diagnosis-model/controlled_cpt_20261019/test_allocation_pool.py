"""Synthetic scheduler fixtures only; no patient inputs or real GPU commands."""
import copy
import subprocess
import unittest
from unittest.mock import patch
from allocation_pool import select_pool, query, SchedulerUnavailable


class Selection(unittest.TestCase):
    def setUp(self):
        self.specs = [dict(lookup_id=str(i), job_id=str(i), node='g-44-01',
                          allowed_idle_steps={'3.0':'sleep'} if i == 3 else {}) for i in (1, 2, 3)]
        self.records = {str(i):dict(JobId=str(i), UserId='research_user(123)', JobState='RUNNING',
                        Partition='gpu-xe9680q', NodeList='g-44-01', AllocTRES='cpu=4,mem=72G,gres/gpu=1')
                        for i in (1, 2, 3)}

    def test_known_sleep_can_coexist_without_stopping_allocation(self):
        pool = select_pool(self.specs, self.records, [('1.batch','batch'), ('3.0','sleep')])
        self.assertEqual(len(pool), 3)
        self.assertEqual([p['step_overlap'] for p in pool], [False, False, True])

    def test_other_work_and_renamed_holder_block_admission(self):
        for steps in [[('1.4','train')], [('3.0','python')], [('3.1','sleep')]]:
            self.assertEqual(len(select_pool(self.specs, self.records, steps)), 2)

    def test_unrelated_jobs_do_not_affect_selected_allocations(self):
        self.assertEqual(len(select_pool(self.specs, self.records, [('99.0','training')])), 3)

    def test_identity_resources_and_duplicate_ids_fail_closed(self):
        for key, value in [('UserId','other(999)'), ('JobId','999'), ('Partition','other'),
                           ('NodeList','different'), ('AllocTRES','cpu=4,mem=32G,gres/gpu=1'),
                           ('AllocTRES','cpu=4,mem=72G,gres/gpu=2')]:
            records = copy.deepcopy(self.records); records['1'][key] = value
            with self.assertRaises(RuntimeError): select_pool(self.specs, records, [])
        specs=copy.deepcopy(self.specs); specs[1]['job_id']='1'
        with self.assertRaises(RuntimeError): select_pool(specs,self.records,[])

    def test_pending_waits_but_cancellation_requires_review(self):
        self.records['1']['JobState']='PENDING'
        self.assertEqual(len(select_pool(self.specs,self.records,[])),2)
        self.records['1']['JobState']='CANCELLED'
        with self.assertRaises(RuntimeError): select_pool(self.specs,self.records,[])

    def test_transient_timeouts_retry_but_invalid_job_fails_closed(self):
        with patch('allocation_pool.subprocess.run',side_effect=subprocess.TimeoutExpired('squeue',30)):
            with self.assertRaises(SchedulerUnavailable): query(['squeue'])
        r=subprocess.CompletedProcess('squeue',1,'','controller unavailable')
        with patch('allocation_pool.subprocess.run',return_value=r):
            with self.assertRaises(SchedulerUnavailable): query(['squeue'])
        r=subprocess.CompletedProcess('scontrol',1,'','Invalid job id specified')
        with patch('allocation_pool.subprocess.run',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'no_longer_available'): query(['scontrol'])


if __name__=='__main__': unittest.main()
