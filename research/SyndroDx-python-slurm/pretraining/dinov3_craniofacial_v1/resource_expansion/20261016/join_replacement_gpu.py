"""Arrange a checkpoint-backed join of an accepted replacement H100."""
import fcntl
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,'/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
from runtime import ROOT, load, save, now, notify


def next_checkpoint(iteration, period=250):
    return ((iteration + 11 + period - 1) // period) * period - 1


def schedule(folder, job):
    deadline = time.monotonic() + 6 * 3600
    request_path = ROOT/'training/checkpoint_batch_handoff.json'
    while time.monotonic() < deadline:
        with (ROOT/'training/checkpoint_batch_apply.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            launch = load(ROOT/'training/launcher.json')
            state = load(ROOT/'training/state.json')
            request = load(request_path)
            if job in launch.get('allocation_job_ids', []):
                save(folder/'join_status.json', dict(state='already_training', job_id=job, updated_at_utc=now()))
                return
            if state.get('state') == 'complete':
                save(folder/'join_status.json',dict(state='available_for_posttraining_evaluation',job_id=job,updated_at_utc=now()))
                return
            if launch.get('state') == 'running' and state.get('state') == 'training' and state.get('world_size') == launch.get('world_size') and state.get('updated_at_utc','') >= launch.get('started_at_utc',''):
                same_attempt = request.get('expected_started_at_utc') == launch.get('started_at_utc')
                if same_attempt and request.get('state') == 'waiting_for_checkpoint':
                    ids = list(dict.fromkeys(request.get('new_allocation_job_ids', []) + [job]))
                    expected = int(launch['world_size']) + len([v for v in ids if v not in launch.get('allocation_job_ids',[])])
                    save(request_path,dict(request,new_allocation_job_ids=ids,expected_training_world_size=expected,purpose='join_accepted_additional_H100_allocations',updated_at_utc=now()))
                    save(folder/'join_status.json',dict(state='added_to_pending_checkpoint_handoff',job_id=job,target_checkpoint_iteration=request['target_checkpoint_iteration'],expected_training_world_size=expected,updated_at_utc=now()))
                    return
                if request.get('state') not in ['waiting_for_checkpoint','applying']:
                    target=next_checkpoint(int(state.get('iteration',state.get('start_iteration',0))))
                    if target >= int(state['total_iterations'])-1:
                        save(folder/'join_status.json',dict(state='available_for_posttraining_evaluation',job_id=job,reason='training_finishes_before_next_regular_checkpoint',updated_at_utc=now()))
                        return
                    save(request_path,dict(state='waiting_for_checkpoint',purpose='join_validation_replacement_H100',target_checkpoint_iteration=target,expected_started_at_utc=launch['started_at_utc'],coordinator_job_id=str(load(ROOT/'coordinator_submission.json')['job_id']),new_allocation_job_ids=[job],expected_training_world_size=int(launch['world_size'])+1,parent_allocations_cancelled=False,retain_per_gpu_batch=64,sample_budget_and_patient_split_unchanged=True,requested_at_utc=now(),updated_at_utc=now()))
                    previous=load(ROOT/'training/checkpoint_batch_submission.json')
                    save(ROOT/'training/checkpoint_batch_submission.json',dict(previous,attempts=0,target_checkpoint_iteration=target))
                    from ensure_checkpoint_batch_watch import ensure
                    ensure()
                    save(folder/'join_status.json',dict(state='checkpoint_handoff_scheduled',job_id=job,target_checkpoint_iteration=target,updated_at_utc=now()))
                    return
        time.sleep(20)
    raise RuntimeError('replacement_checkpoint_join_timeout')


if __name__ == '__main__':
    folder=Path(sys.argv[1])
    try:
        if load(folder/'ready.json').get('state') != 'passed':
            raise RuntimeError('replacement_acceptance_not_passed')
        schedule(folder,os.environ['SLURM_JOB_ID'])
    except Exception as exc:
        save(folder/'join_status.json',dict(state='requires_attention',error_type=type(exc).__name__,updated_at_utc=now()))
        notify('Replacement H100 join requires attention','SSL GPU expansion requires attention','A replacement GPU could not schedule its checkpoint handoff. Existing training and checkpoints are unchanged. Private data remain on HPC.')
        # Preserve the reservation; the existing controller also discovers it
        # automatically at any subsequent natural segment boundary.
