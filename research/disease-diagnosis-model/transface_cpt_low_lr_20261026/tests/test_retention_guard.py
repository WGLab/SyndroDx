"""Verify checkpoint protection and non-stopping validation policy."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import retention
from guard_policy import advisory
from control import save,digest

class RecoveryTests(unittest.TestCase):
    def test_early_validation_and_best_weights_are_protected(self):
        steps=list(range(0,4050,50))
        keep=retention.protected_steps(steps,dict(validation_first_steps=[100,250],validation_every_steps=250),[2500])
        self.assertTrue({0,100,250,500,1000,1500,2000,2500,3250,3500,3750,4000,3900,3950}<=keep)
        self.assertNotIn(2050,keep)

    def test_best_snapshot_survives_source_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(retention,'WORK',Path(folder)):
            root=Path(folder);(root/'checkpoints').mkdir();(root/'validation').mkdir();(root/'best').mkdir()
            path=root/'checkpoints/step_0000250.pt';path.write_bytes(b'synthetic checkpoint')
            save(path.with_suffix('.json'),dict(state='complete',sha256=digest(path)))
            save(root/'validation/step_0000250.json',dict(state='complete',metrics={k:{v:{'top1':.3} for v in ('ridge','retrieval')} for k in ('disease','structural')}))
            retention.record_validation(250);path.unlink()
            self.assertEqual((root/'best/disease_retrieval.pt').read_bytes(),b'synthetic checkpoint')

    def test_metric_decline_reduces_lr_without_stopping(self):
        d=advisory(False,{'disease'},1.)
        self.assertTrue(d['continue_training']);self.assertEqual(d['multiplier'],.5)
        self.assertEqual(advisory(True,set(),.01)['multiplier'],.01)
        self.assertEqual(advisory(False,set(),.5)['multiplier'],.5)

if __name__=='__main__':unittest.main()
