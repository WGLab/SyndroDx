import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import workflow
from common import save,digest

class HandoffTests(unittest.TestCase):
    def test_monitor_restart_preserves_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);save(work/'results.json',dict(state='complete',updated_at_utc='2026-10-27T07:48:16+00:00'))
            before=(work/'results.json').read_bytes()
            with patch.object(workflow,'WORK',work),patch.object(workflow,'completed_heartbeat') as heartbeat:
                workflow.finish();heartbeat.assert_called_once()
            self.assertEqual((work/'results.json').read_bytes(),before)
    def test_follow_on_updates_heartbeat_watched_by_gpu_leases(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'follow';leases=Path(tmp)/'leases'
            with patch.object(workflow,'WORK',work),patch.object(workflow,'LEASES',leases):
                workflow.heartbeat('monitoring_identity_cpt')
                self.assertFalse((leases/'supervisor_heartbeat.json').exists())
                workflow.heartbeat('disease_adaptation')
                receipt=json.loads((leases/'supervisor_heartbeat.json').read_text())
                self.assertEqual(receipt['owner'],'follow')
                workflow.heartbeat('complete',allocations_retained=False)
                self.assertFalse(json.loads((leases/'supervisor_heartbeat.json').read_text())['allocations_retained'])
    def test_missing_allocations_are_not_reported_as_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            leases=Path(tmp);save(leases/'pool_registry.json',dict(jobs={'0':'123','1':'124'}))
            with patch.object(workflow,'LEASES',leases),patch.object(workflow,'command',return_value=''):
                result=workflow.allocation_snapshot()
            self.assertEqual(result['allocated_gpus'],0);self.assertEqual(result['missing_gpu_leases'],2)
            self.assertFalse(result['allocations_retained'])
    def test_fallback_yields_only_to_registered_slurm_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);save(work/'workflow_job.json',dict(job_id='123'))
            save(work/'slurm_coordinator_claim.json',dict(job_id='456',updated_at_utc=workflow.now()))
            with patch.object(workflow,'WORK',work),patch.object(workflow,'RUNNING_IN_SLURM',False):
                workflow.yield_to_slurm()
                save(work/'slurm_coordinator_claim.json',dict(job_id='123',updated_at_utc=workflow.now()))
                with self.assertRaises(workflow.CoordinatorHandoff):workflow.yield_to_slurm()
    def test_checkpoint_handoff_pins_best_validated_not_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'stage';work=Path(tmp)/'next'
            (stage/'checkpoints').mkdir(parents=True);(work/'selected').mkdir(parents=True)
            old=stage/'best.pt';old.write_bytes(b'synthetic validated full checkpoint')
            new=stage/'latest.pt';new.write_bytes(b'synthetic later checkpoint')
            save(stage/'best/registry.json',dict(disease_ridge=dict(step=750,macro_top1=.25,path=str(old),sha256=digest(old))))
            save(stage/'latest_checkpoint.json',dict(step=3000,state='complete',path=str(new),sha256=digest(new)))
            with patch.object(workflow,'WORK',work),patch.object(workflow,'STAGE1',stage):workflow.select_identity_checkpoint()
            chosen=json.loads((work/'selected/identity_checkpoint.json').read_text())
            self.assertEqual(chosen['step'],750);self.assertEqual(digest(chosen['path']),digest(old))
    def test_checksum_failure_prevents_stage_advancement(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'stage';work=Path(tmp)/'next'
            (stage/'checkpoints').mkdir(parents=True);(work/'selected').mkdir(parents=True)
            ckpt=stage/'latest.pt';ckpt.write_bytes(b'synthetic corruption')
            save(stage/'latest_checkpoint.json',dict(step=100,path=str(ckpt),sha256='bad'))
            with patch.object(workflow,'WORK',work),patch.object(workflow,'STAGE1',stage):
                with self.assertRaisesRegex(RuntimeError,'checksum'):workflow.select_identity_checkpoint()
            self.assertFalse((work/'selected/identity_checkpoint.json').exists())
    def test_matched_selection_requires_all_cases_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(workflow,'WORK',Path(tmp)):
                with self.assertRaisesRegex(RuntimeError,'incomplete'):workflow.select_disease_models([])
            self.assertFalse((Path(tmp)/'selected/models.json').exists())
    def test_finished_task_not_launched_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);save(work/'cases/synthetic/result.json',dict(state='complete'))
            task=dict(key='synthetic',name='synthetic',folder='cases/synthetic')
            with patch.object(workflow,'WORK',work),patch.object(workflow,'verify_code'),patch.object(workflow,'stage1_modules') as leases,patch.object(workflow,'ready_pool',return_value=[]),patch.object(workflow,'step_table',return_value=[]),patch.object(workflow,'launch') as launch:
                self.assertTrue(workflow.run_tasks([task],'test'));launch.assert_not_called();leases.assert_not_called()

if __name__=='__main__':unittest.main()
