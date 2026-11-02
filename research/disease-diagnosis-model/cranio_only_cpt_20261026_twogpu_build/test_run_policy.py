"""Synthetic duration, checkpoint recovery and duplicate-free lease renewal tests."""
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from run_policy import ActiveTrainingClock,completion,validation_policy,MINIMUM_TRAINING_SECONDS
from common import load,save,digest
from checkpoint_recovery import recover,valid_checkpoint
import lease_supervisor as supervisor


class DurationTests(unittest.TestCase):
    def test_early_plateau_never_completes(self):
        for seconds in [0,3600,86400,MINIMUM_TRAINING_SECONDS-1]:
            self.assertFalse(completion(seconds,True)['complete'])
    def test_ten_days_without_plateau_continues(self):
        self.assertFalse(completion(MINIMUM_TRAINING_SECONDS,False)['complete'])
        self.assertTrue(completion(MINIMUM_TRAINING_SECONDS,True)['complete'])
    def test_queue_validation_and_restarts_are_not_training_time(self):
        now=[0.];clock=ActiveTrainingClock(123.,timer=lambda:now[0])
        now[0]=1000.;self.assertEqual(clock.pause(),123.)
        clock.start();clock.start();now[0]=1010.;self.assertEqual(clock.pause(),133.)
        now[0]=9000.;self.assertEqual(clock.pause(),133.)
        resumed=ActiveTrainingClock(clock.credited,timer=lambda:now[0]);resumed.start()
        now[0]=9007.;self.assertEqual(resumed.pause(),140.)
    def test_invalid_duration_rejected(self):
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):completion(value,True)
    def test_regression_warns_but_health_failure_does_not_pass(self):
        self.assertTrue(validation_policy(dict(state='paused_validation_regression',continue_training=False))['continue_training'])
        for state in ['paused_invalid_validation','paused_feature_collapse']:
            self.assertFalse(validation_policy(dict(state=state,continue_training=False))['continue_training'])


class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        save(self.root/'resume_plan.json',dict(completed_steps=5000,source_world=4,consumed_global_samples=640000))
        p=self.root/'source_snapshot/ckpt/4999';p.mkdir(parents=True);(p/'.metadata').write_bytes(b'source')
        save(self.root/'attempts/000.json',dict(arm='stage0'))
        self.folder=self.root/'stage0/training'
    def tearDown(self):self.tmp.cleanup()
    def boundary(self,step,seconds):
        ckpt=self.folder/'ckpt'/str(step-1);ckpt.mkdir(parents=True)
        (ckpt/'.metadata').write_bytes(b'metadata');(ckpt/'shard').write_bytes(b'full student teacher optimizer')
        (self.folder/'config.yaml').write_text('synthetic: true\n')
        report=self.folder/'validation'/f'step_{step:06d}'
        save(report/'report.json',dict(state='complete',decision=dict(continue_training=True)))
        for rank in range(4):(report/f'rank_{rank:02d}.private.npy').write_bytes(b'synthetic features')
        row=dict(completed_steps=step,source_world=4,consumed_global_samples=640000+(step-5000)*128,
            training_seconds_at_resume=seconds,metadata_sha256=digest(ckpt/'.metadata'),
            checkpoint_file_sizes={p.name:p.stat().st_size for p in ckpt.iterdir()})
        save(self.folder/'checkpoint_ready'/f'step_{step:06d}.json',row)
        return row
    def test_crash_recovers_latest_complete_checkpoint_and_clock(self):
        self.boundary(5100,123.);self.boundary(5200,245.)
        state,attempt=recover(self.root)
        self.assertEqual(state['completed_steps'],5200);self.assertEqual(state['training_seconds_at_resume'],245.)
        self.assertEqual(attempt,1)
        self.assertEqual(state['consumed_global_samples'],665600)
        state2,_=recover(self.root);self.assertEqual(state2,state)
    def test_partial_checkpoint_never_selected(self):
        self.boundary(5100,123.);row=self.boundary(5200,245.)
        (self.folder/'ckpt/5199/shard').write_bytes(b'truncated')
        self.assertFalse(valid_checkpoint(self.folder,row))
        state,_=recover(self.root);self.assertEqual(state['completed_steps'],5100)
    def test_committed_checkpoint_is_rechecked_before_resume(self):
        self.boundary(5100,123.)
        state,_=recover(self.root)
        (Path(state['source_training'])/'ckpt/5099/shard').write_bytes(b'truncated')
        with self.assertRaisesRegex(RuntimeError,'integrity_failed'):recover(self.root)
    def test_failed_validation_never_selected(self):
        self.boundary(5100,123.)
        save(self.folder/'validation/step_005100/report.json',dict(decision=dict(continue_training=False)))
        state,_=recover(self.root);self.assertEqual(state['completed_steps'],5000)
        self.assertEqual(state['training_seconds_at_resume'],0.)
    def test_no_update_does_not_overwrite_existing_attempt(self):
        state,attempt=recover(self.root)
        self.assertEqual(state['completed_steps'],5000);self.assertEqual(attempt,1)


class LeaseRenewalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.jobs={str(i):str(100+i) for i in range(20)}
        save(self.root/'pool_registry.json',dict(jobs=self.jobs))
        self.current={job:dict(name='existing-'+slot,state='PENDING',node='') for slot,job in self.jobs.items()}
        self.scope=patch.object(supervisor,'WORK',self.root);self.scope.start()
    def tearDown(self):self.scope.stop();self.tmp.cleanup()
    def test_pending_and_running_jobs_are_preserved(self):
        self.current['100']['state']='RUNNING'
        with patch.object(supervisor,'command') as call:
            supervisor.reconcile_pool(self.current)
            call.assert_not_called()
        self.assertEqual(load(self.root/'pool_registry.json')['jobs'],self.jobs)
    def test_missing_job_without_terminal_proof_not_replaced(self):
        self.current.pop('100')
        with patch.object(supervisor,'command',return_value='') as call:
            supervisor.reconcile_pool(self.current)
            self.assertEqual(call.call_count,1)
        self.assertEqual(load(self.root/'pool_registry.json')['jobs']['0'],'100')
    def test_terminal_job_replaced_once_remaining_nineteen_preserved(self):
        self.current.pop('100')
        def response(args,**kwargs):
            if args[0]=='sacct':return '100|TIMEOUT\n'
            if args[0]=='sbatch':return '500\n'
            raise AssertionError(args)
        with patch.object(supervisor,'command',side_effect=response),patch('pool_work.release_owner_after_exit'):
            supervisor.reconcile_pool(self.current)
        result=load(self.root/'pool_registry.json')['jobs']
        self.assertEqual(result['0'],'500');self.assertEqual(len(result),20)
        self.assertEqual({k:v for k,v in result.items() if k!='0'},{k:v for k,v in self.jobs.items() if k!='0'})
    def test_ambiguous_acknowledgement_is_not_resubmitted(self):
        self.current.pop('100');name=supervisor.replacement_name('0','100')
        save(self.root/'replacement_journal.json',{name:dict(attempted=True,started_at_utc='2026-10-25T00:00:00+00:00')})
        calls=[]
        def response(args,**kwargs):
            calls.append(args)
            return '100|TIMEOUT\n' if '-j' in args else ''
        with patch.object(supervisor,'command',side_effect=response):supervisor.reconcile_pool(self.current)
        self.assertFalse(any(a[0]=='sbatch' for a in calls))
    def test_accepted_job_reconciled_by_name_after_supervisor_crash(self):
        self.current.pop('100');name=supervisor.replacement_name('0','100')
        self.current['500']=dict(name=name,state='PENDING',node='')
        with patch.object(supervisor,'command',return_value='100|TIMEOUT\n') as call,patch('pool_work.release_owner_after_exit'):
            supervisor.reconcile_pool(self.current)
        self.assertEqual(call.call_count,1)
        self.assertEqual(load(self.root/'pool_registry.json')['jobs']['0'],'500')
    def test_cleanup_never_cancels_allocations_or_unrelated_steps(self):
        calls=[]
        def response(args,**kwargs):
            calls.append(args)
            return '100.0|cranio-pool-train\n100.batch|batch\n999.0|cranio-pool-train\n101.3|another-task\n' if args[0]=='squeue' else ''
        with patch.object(supervisor,'command',side_effect=response):self.assertFalse(supervisor.cleanup_steps())
        self.assertEqual(calls[-1],['scancel','100.0'])
    def test_pid_reuse_cannot_kill_an_unrelated_process(self):
        with patch.object(supervisor,'start_identity',return_value='different'),patch.object(supervisor.os,'killpg') as kill:
            supervisor.stop_group(dict(pid=123,start_identity='original'));kill.assert_not_called()
    def test_process_identity_is_live_and_host_scoped(self):
        identity=supervisor.start_identity(os.getpid())
        self.assertIsNotNone(identity)
        self.assertTrue(supervisor.same_process(dict(pid=os.getpid(),start_identity=identity)))
        self.assertFalse(supervisor.same_process(dict(pid=os.getpid(),start_identity=identity,host='a-different-node')))
        with patch.object(supervisor.os,'killpg') as kill:
            supervisor.stop_group(dict(pid=os.getpid(),start_identity=identity,host='a-different-node'))
            kill.assert_not_called()


if __name__=='__main__':unittest.main()
