"""Synthetic operational plateau and continued-budget regression checks."""
import copy
import unittest
from convergence import assess,COMPONENTS


def records(step,old=10.,new=10.):
    result=[]
    for iteration in range(step-1000,step,10):
        loss=old if iteration<step-500 else new
        result.append(dict(iteration=iteration,total_loss=loss,lr=5e-5,
            **{k:loss/3 for k in COMPONENTS}))
    return result


class ConvergenceTests(unittest.TestCase):
    def test_still_falling_does_not_stop_at_arbitrary_budget(self):
        report=assess(records(6000,14.,13.8),6000,5000)
        self.assertFalse(report['plateau']);self.assertEqual(report['state'],'loss_still_changing')
    def test_three_distinct_stable_checks_required(self):
        previous={}
        for step in (6000,6250,6500):
            previous=assess(records(step,14.,13.99),step,5000,previous)
            self.assertEqual(previous['plateau'],step==6500)
        self.assertTrue(previous['not_a_representation_quality_claim'])
    def test_cannot_call_converged_after_resize_with_short_window(self):
        self.assertFalse(assess(records(6500),6500,6000)['plateau'])
    def test_component_still_moving_prevents_total_loss_cancellation(self):
        values=records(6000)
        for r in values[50:]:r['ibot_loss']*=.98
        self.assertEqual(assess(values,6000,5000)['state'],'loss_still_changing')
    def test_noisy_loss_and_increasing_loss_do_not_pass(self):
        values=records(6000)
        for i,r in enumerate(values[50:]):r['total_loss']+=1 if i%2 else -1
        self.assertFalse(assess(values,6000,5000)['plateau'])
        self.assertEqual(assess(records(6000,10.,11.),6000,5000)['state'],'loss_still_changing')
    def test_missing_nan_and_changed_lr_are_not_plateau(self):
        values=records(6000);values[-1]['ibot_loss']=float('nan')
        self.assertEqual(assess(values,6000,5000)['state'],'invalid_convergence_metrics')
        values=records(6000);values[-1]['lr']=1e-5
        self.assertEqual(assess(values,6000,5000)['state'],'learning_rate_changed_within_window')
        self.assertEqual(assess(records(6000)[:20],6000,5000)['state'],'incomplete_convergence_window')
    def test_nonconsecutive_checks_do_not_accumulate(self):
        first=assess(records(6000),6000,5000)
        later=assess(records(6500),6500,5000,first)
        self.assertEqual(later['consecutive_stable_checks'],1)


if __name__=='__main__':unittest.main()
