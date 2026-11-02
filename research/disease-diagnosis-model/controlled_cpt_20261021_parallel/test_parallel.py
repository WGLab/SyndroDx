"""Synthetic scheduling and paired-integrity failure-path tests."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from configuration import WORLD, ARMS
from allocation_pool import select_pools
from controller import audit_inputs
from common import save


class Pools(unittest.TestCase):
    def setUp(self):
        self.specs, self.records, self.steps = {}, {}, []
        for ai, arm in enumerate(ARMS):
            self.specs[arm] = []
            for rank in range(WORLD):
                j = str(100+ai*WORLD+rank)
                node = 'node'+str(ai)
                self.specs[arm].append(dict(job_id=j, lookup_id=j, node=node,
                    host_memory_gib=68, allowed_idle_steps={j+'.0':'sleep'}))
                self.records[j] = dict(JobId=j, UserId='research_user(1)', JobState='RUNNING',
                    NodeList=node, Partition='gpu-xe9680q', AllocTRES='cpu=4,mem=72G,gres/gpu=1')
                self.steps.append((j+'.0','sleep'))

    def test_two_disjoint_pools_and_only_known_holders(self):
        result=select_pools(self.specs,self.records,self.steps)
        self.assertEqual([len(result[a]) for a in ARMS], [8,8])
        self.assertTrue(all(p['step_overlap'] for a in ARMS for p in result[a]))

    def test_duplicate_lease_rejected(self):
        self.specs[ARMS[1]][0]['job_id']='100'
        with self.assertRaisesRegex(RuntimeError,'overlapping'): select_pools(self.specs,self.records,self.steps)

    def test_compute_or_unknown_sleep_blocks_admission(self):
        for name in ('python','sleep'):
            result=select_pools(self.specs,self.records,self.steps+[('100.4',name)])
            self.assertEqual(len(result[ARMS[0]]),7)

    def test_wrong_owner_node_or_insufficient_ram_rejected(self):
        for key,value in [('UserId','other(2)'),('NodeList','wrong'),('AllocTRES','cpu=4,mem=32G,gres/gpu=1')]:
            records=copy.deepcopy(self.records); records['100'][key]=value
            with self.assertRaises(RuntimeError): select_pools(self.specs,records,self.steps)

    def test_cancelled_lease_fails_and_pending_waits(self):
        self.records['100']['JobState']='PENDING'
        self.assertEqual(len(select_pools(self.specs,self.records,self.steps)[ARMS[0]]),7)
        self.records['100']['JobState']='CANCELLED'
        with self.assertRaises(RuntimeError): select_pools(self.specs,self.records,self.steps)


class PairedAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        for arm in ARMS:
            for rank in range(WORLD):
                save(self.path(arm,rank), {'sha256':'synthetic-'+str(rank)})
        self.patch=patch('controller.WORK',self.root);self.patch.start()

    def path(self,arm,rank):
        return self.root/arm/'training/input_hashes'/f'rank_{rank:02d}_step_000001.json'

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_matching_common_prefix(self):
        self.assertEqual(audit_inputs(1)['rank_batches_checked'],8)

    def test_any_missing_or_mismatched_rank_is_fatal(self):
        save(self.path(ARMS[1],7),{'sha256':'different'})
        with self.assertRaises(RuntimeError):audit_inputs(1)
        self.path(ARMS[1],7).unlink()
        with self.assertRaises(RuntimeError):audit_inputs(1)


if __name__=='__main__':unittest.main()
