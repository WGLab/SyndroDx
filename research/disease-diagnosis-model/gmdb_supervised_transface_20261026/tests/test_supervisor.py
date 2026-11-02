import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'transface_arcface_identity_cpt_20261026'))
from supervisor import handoff_ready

class CheckpointHandoff(unittest.TestCase):
    def test_previous_stop_receipt_cannot_retire_active_training(self):
        self.assertFalse(handoff_ready(dict(state='checkpoint_preserved_allocations_retained',updated_at_utc='2026-10-27T02:00:00'),dict(requested_at_utc='2026-10-27T04:00:00')))
    def test_completed_current_stop_allows_handoff(self):
        self.assertTrue(handoff_ready(dict(state='checkpoint_preserved_allocations_retained',updated_at_utc='2026-10-27T04:00:01'),dict(requested_at_utc='2026-10-27T04:00:00')))

if __name__=='__main__':unittest.main()
