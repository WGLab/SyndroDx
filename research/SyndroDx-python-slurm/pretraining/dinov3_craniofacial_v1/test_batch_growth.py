import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
import automatic_batch_tuning as tuning
from test_automatic_batch_tuning import BoundaryHandoff


class BatchGrowth(unittest.TestCase):
    def run_profile(self, memory_at_112):
        setup = BoundaryHandoff()
        setup.setUp()
        try:
            setup.write('batch_tuning_request.json', dict(state='pending', candidate_batches_per_gpu=[96, 112, 128]))
            def profile(selected, config, batch, output):
                if batch == 128:
                    return dict(state='failed', batch=batch)
                return dict(state='passed', all_ranks_passed=True, batch=batch,
                            median_step_seconds={64: 27, 96: 36, 112: 40}[batch],
                            peak_reserved_GiB={64: 48, 96: 65, 112: memory_at_112}[batch])
            with patch.object(tuning, 'benchmark', side_effect=profile) as probe:
                tuning.prepare(setup.jobs, setup.cfg)
            return OmegaConf.load(setup.cfg).train.batch_size_per_gpu, probe.call_count
        finally:
            setup.tearDown()

    def test_fastest_safe_batch_selected_and_oom_retains_checkpoint(self):
        batch, calls = self.run_profile(70)
        self.assertEqual(batch, 112)
        self.assertEqual(calls, 4)

    def test_no_larger_probe_after_memory_limit(self):
        batch, calls = self.run_profile(75)
        self.assertEqual(batch, 96)
        self.assertEqual(calls, 3)


if __name__ == '__main__':
    unittest.main()
