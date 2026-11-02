import math
import unittest
from batch_policy import rebase, progress, schedule_index, sampler_advance, select_batch


class ExposureBudget(unittest.TestCase):
    def test_original_and_increased_batch_have_equal_sample_budget(self):
        policy = rebase({}, 1000, 96, 17, 2248, 1088)
        self.assertEqual(policy['total_iterations'], 1832)
        self.assertEqual(progress(policy, 1000), 1000)
        self.assertEqual(progress(policy, 1832), 2248)
        self.assertEqual(sampler_advance(policy, 1000, 17), 64000)
        self.assertEqual(schedule_index(policy, 1000, 2248), 1000)
        self.assertEqual(schedule_index(policy, 1001, 2248), 1001)
        self.assertLess(schedule_index(policy, 1831, 2248), 2248)

    def test_checkpoint_resume_and_oom_fallback_preserve_progress(self):
        policy = rebase({}, 1000, 96, 17, 2248, 1088)
        fallback = rebase(policy, 1250, 48, 17, 2248, 1088)
        self.assertEqual(progress(fallback, 1250), progress(policy, 1250))
        self.assertGreaterEqual(progress(fallback, fallback['total_iterations']), 2248)
        self.assertLess(progress(fallback, fallback['total_iterations'] - 1), 2248)

    def test_world_change_and_rounding_do_not_drop_budget(self):
        policy = rebase({}, 1000, 96, 16, 2248, 1088)
        self.assertGreaterEqual(progress(policy, policy['total_iterations']), 2248)
        self.assertLess(progress(policy, policy['total_iterations'] - 1), 2248)
        again = rebase(policy, 1100, 96, 17, 2248, 1088)
        self.assertEqual(progress(again, 1100), progress(policy, 1100))

    def test_no_policy_matches_existing_scheduler(self):
        self.assertEqual(progress({}, 42), 42)
        self.assertEqual(schedule_index({}, 42, 2248), 42)


class ThroughputSelection(unittest.TestCase):
    def profile(self, batch, seconds, **kw):
        return dict(state='passed', all_ranks_passed=True, batch=batch,
                    median_step_seconds=seconds, peak_reserved_GiB=kw.get('memory', 65))

    def test_requires_measured_speedup_and_headroom(self):
        base = self.profile(64, 27)
        self.assertEqual(select_batch([base, self.profile(96, 34)], 64), 96)
        self.assertEqual(select_batch([base, self.profile(96, 40)], 64), 64)
        self.assertEqual(select_batch([base, self.profile(96, 34, memory=75)], 64), 64)
        self.assertEqual(select_batch([dict(base, state='failed'), self.profile(96, 34)], 64), 64)
        self.assertEqual(select_batch([base, dict(self.profile(96, 34), all_ranks_passed=False)], 64), 64)


if __name__ == '__main__':
    unittest.main()
