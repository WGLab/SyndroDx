import datetime
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy import decide,choose_checkpoint
from common import WORK,load

class StopPolicy(unittest.TestCase):
    def setUp(self):
        self.policy=load(WORK/'protocol.json')['stop_policy']
        self.now=datetime.datetime(2026,10,27,tzinfo=datetime.timezone.utc)
        self.rows=[dict(completed_steps=i,loss=1.,world_size=8,updated_at_utc=self.now.isoformat()) for i in range(10,4001,10)]
        self.reports=[dict(state='complete',completed_steps=i,metrics={'disease':{'ridge':{'top1':.25}}}) for i in range(0,4001,250)]
    def test_consistent_loss_and_development_plateau_advances(self):
        self.assertTrue(decide(self.rows,self.reports,self.policy,self.now)['advance'])
    def test_decreasing_loss_continues(self):
        for r in self.rows:r['loss']=10-r['completed_steps']/1000
        self.assertFalse(decide(self.rows,self.reports,self.policy,self.now)['advance'])
    def test_development_improvement_continues(self):
        self.reports[-1]['metrics']['disease']['ridge']['top1']=.30
        self.assertFalse(decide(self.rows,self.reports,self.policy,self.now)['advance'])
    def test_resize_at_window_boundary_not_mistaken_for_convergence(self):
        for r in self.rows:r['world_size']=8 if r['completed_steps']<=3000 else 16
        self.assertFalse(decide(self.rows,self.reports,self.policy,self.now)['advance'])
    def test_stale_development_cannot_advance(self):
        self.assertFalse(decide(self.rows,self.reports[:10],self.policy,self.now)['advance'])
    def test_budget_is_not_convergence(self):
        result=decide(self.rows,self.reports,self.policy,self.now+datetime.timedelta(days=11))
        self.assertTrue(result['advance']);self.assertIn('not_claimed_converged',result['reason'])
    def test_no_early_stop(self):
        self.assertFalse(decide(self.rows[:200],self.reports,self.policy,self.now)['advance'])
    def test_incomplete_or_zero_checkpoint_never_selected(self):
        candidates=[dict(step=0,score=1.,complete=True),dict(step=100,score=.8,complete=False),dict(step=200,score=.2,complete=True)]
        self.assertEqual(choose_checkpoint(candidates)['step'],200)

if __name__=='__main__':unittest.main()
