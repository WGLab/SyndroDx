import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import automatic_batch_tuning as tuning


class BoundaryHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / 'training'
        self.output.mkdir()
        self.cfg = self.output / 'run_config.yaml'
        OmegaConf.save(OmegaConf.create({'train': {'batch_size_per_gpu': 64, 'OFFICIAL_EPOCH_LENGTH': 1},
                                        'optim': {'epochs': 2248}}), self.cfg)
        self.write('run_plan.json', {'initial_world_size': 17, 'batch_size_per_gpu': 64})
        self.write('batch_tuning_request.json', {'state': 'pending', 'candidate_batch_size_per_gpu': 96})
        checkpoint = self.output / 'ckpt/999'
        checkpoint.mkdir(parents=True)
        (checkpoint / '.metadata').touch()
        self.jobs = [{'job_id': str(i), 'gpus': 1, 'node': 'synthetic-node'} for i in range(17)]
        self.root_patch = patch.object(tuning, 'ROOT', self.root)
        self.root_patch.start()
        self.notify_patch = patch.object(tuning, 'notify')
        self.notify_patch.start()

    def tearDown(self):
        self.notify_patch.stop()
        self.root_patch.stop()
        self.tmp.cleanup()

    def write(self, name, value):
        (self.output / name).write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.output / name).read_text())

    def profile(self, selected, config, batch, output):
        return dict(state='passed', all_ranks_passed=True, batch=batch,
                    median_step_seconds=27 if batch == 64 else 34, peak_reserved_GiB=65)

    def test_safe_profile_increases_batch_and_preserves_original_schedules(self):
        with patch.object(tuning, 'benchmark', side_effect=self.profile) as benchmark:
            tuning.prepare(self.jobs, self.cfg)
            self.assertEqual(benchmark.call_count, 2)
        cfg = OmegaConf.load(self.cfg)
        self.assertEqual(cfg.train.batch_size_per_gpu, 96)
        self.assertEqual(cfg.optim.epochs, 2248)
        policy = self.read('execution_policy.json')
        self.assertEqual(policy['total_iterations'], 1832)
        self.assertEqual(policy['anchor_iteration'], 1000)
        self.assertEqual(self.read('batch_tuning_request.json')['state'], 'complete')

    def test_candidate_failure_retains_batch(self):
        def profile(*args):
            return self.profile(*args) if args[2] == 64 else dict(state='failed', batch=96)
        with patch.object(tuning, 'benchmark', side_effect=profile):
            tuning.prepare(self.jobs, self.cfg)
        self.assertEqual(OmegaConf.load(self.cfg).train.batch_size_per_gpu, 64)
        self.assertEqual(self.read('execution_policy.json')['total_iterations'], 2248)

    def test_uncommitted_checkpoint_cannot_trigger_batch_change(self):
        (self.output / 'ckpt/999/.metadata').unlink()
        with patch.object(tuning, 'benchmark') as benchmark:
            with self.assertRaisesRegex(RuntimeError, 'committed_checkpoint_required'):
                tuning.prepare(self.jobs, self.cfg)
            benchmark.assert_not_called()

    def test_next_segment_rebases_without_benchmarking_again(self):
        with patch.object(tuning, 'benchmark', side_effect=self.profile):
            tuning.prepare(self.jobs, self.cfg)
        checkpoint = self.output / 'ckpt/1249'
        checkpoint.mkdir()
        (checkpoint / '.metadata').touch()
        with patch.object(tuning, 'benchmark') as benchmark:
            tuning.prepare(self.jobs, self.cfg)
            benchmark.assert_not_called()
        policy = self.read('execution_policy.json')
        self.assertEqual(policy['anchor_progress'], 1375)
        self.assertEqual(policy['total_iterations'], 1832)

    def test_selected_transaction_resumes_without_rebenchmarking(self):
        self.write('batch_tuning_request.json', {'state': 'selected', 'selected_batch_size_per_gpu': 96})
        with patch.object(tuning, 'benchmark') as benchmark:
            tuning.prepare(self.jobs, self.cfg)
            benchmark.assert_not_called()
        self.assertEqual(OmegaConf.load(self.cfg).train.batch_size_per_gpu, 96)


if __name__ == '__main__':
    unittest.main()
