import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import checkpoint_batch_handoff as handoff
from runtime import load, save


class CheckpointHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / 'training'
        self.output.mkdir()
        self.request = self.output / 'checkpoint_batch_handoff.json'
        save(self.request, dict(state='waiting_for_checkpoint', target_checkpoint_iteration=249,
             expected_started_at_utc='synthetic-attempt', coordinator_job_id='123'))
        save(self.output / 'launcher.json', dict(state='running', world_size=2, started_at_utc='synthetic-attempt',
             launcher_pids=[1001, 1002], allocation_job_ids=['11', '12']))
        save(self.output / 'controller_launch.json', dict(pid=1000, failures=0, launched_at=0))
        save(self.output / 'state.json', dict(state='training', iteration=240, total_iterations=2248))
        self.patches = [patch.object(handoff, 'ROOT', self.root), patch.object(handoff, 'REQUEST', self.request),
                        patch.object(handoff, 'notify'), patch.object(handoff.os, 'killpg')]
        self.mocks = [p.start() for p in self.patches]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def checkpoint(self, iteration=249, complete=True):
        folder = self.output / 'ckpt' / str(iteration)
        folder.mkdir(parents=True)
        for rank in range(2):
            (folder / f'{rank}.distcp').write_bytes(b'synthetic')
        if complete:
            (folder / '.metadata').write_bytes(b'synthetic')

    def test_partial_checkpoint_never_stops_training(self):
        self.checkpoint(complete=False)
        with self.assertRaisesRegex(RuntimeError, 'target_checkpoint_not_committed'):
            handoff.apply()
        self.mocks[-1].assert_not_called()

    def test_old_early_checkpoint_does_not_trigger_handoff(self):
        self.checkpoint(9)
        with self.assertRaisesRegex(RuntimeError, 'target_checkpoint_not_committed'):
            handoff.apply()
        self.mocks[-1].assert_not_called()

    def test_unverified_process_never_stopped(self):
        self.checkpoint()
        with patch.object(handoff, 'verified', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'clients_not_all_verified'):
                handoff.apply()
        self.mocks[-1].assert_not_called()

    def test_planned_handoff_resumes_next_iteration_and_retains_allocations(self):
        self.checkpoint()
        # Verify two clients and their launcher; then all are observed exited.
        with patch.object(handoff, 'verified', side_effect=[True, True, True, False, False, False]):
            handoff.apply()
        self.assertEqual(self.mocks[-1].call_count, 2)
        request = load(self.request)
        self.assertEqual(request['state'], 'restart_enabled')
        self.assertFalse(request['parent_allocations_cancelled'])
        self.assertEqual(load(self.output / 'state.json')['next_iteration'], 250)
        self.assertEqual(load(self.output / 'launcher.json')['state'], 'segment_complete')
        self.assertEqual(load(self.output / 'controller_launch.json')['failures'], 0)
        self.assertTrue((self.output / 'ckpt/249/.metadata').is_file())

    def test_changed_attempt_is_not_stopped(self):
        self.checkpoint()
        launcher = load(self.output / 'launcher.json')
        save(self.output / 'launcher.json', dict(launcher, started_at_utc='new-attempt'))
        handoff.apply()
        self.mocks[-1].assert_not_called()
        self.assertEqual(load(self.request)['state'], 'superseded_by_other_training_attempt')


if __name__ == '__main__':
    unittest.main()
