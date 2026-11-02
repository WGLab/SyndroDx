"""Recover only validated, fully written local checkpoints; no record export."""
from pathlib import Path
from common import load,save,digest,now
from incremental import snapshot


def valid_checkpoint(folder, row):
    step=row.get('completed_steps');world=row.get('source_world')
    if not isinstance(step,int) or step<=0 or not isinstance(world,int) or world<1:return False
    ckpt=folder/'ckpt'/str(step-1)
    if not (ckpt/'.metadata').is_file():return False
    if digest(ckpt/'.metadata')!=row.get('metadata_sha256'):return False
    sizes=row.get('checkpoint_file_sizes',{})
    if len(sizes)<2:return False
    for name,size in sizes.items():
        if Path(name).name!=name or not isinstance(size,int) or size<=0:return False
        path=ckpt/name
        if not path.is_file() or path.stat().st_size!=size:return False
    validation=folder/'validation'/f'step_{step:06d}'
    if not load(validation/'report.json').get('decision',{}).get('continue_training'):return False
    return all((validation/f'rank_{r:02d}.private.npy').is_file() for r in range(world))


def pin(work,folder,row,attempt):
    if not valid_checkpoint(folder,row):raise RuntimeError('checkpoint_not_verified_for_recovery')
    target=work/'snapshots'/f'stage_{attempt:03d}_step_{row["completed_steps"]:06d}'
    if target.exists():
        receipt=load(target/'snapshot_receipt.json')
        if receipt.get('state')!='passed' or not valid_checkpoint(target,row):
            # Keep an interrupted snapshot for diagnosis; never overwrite it.
            import uuid
            target=target.with_name(target.name+'_'+uuid.uuid4().hex[:8])
    if not target.exists():snapshot(folder,target,row['completed_steps'],row['source_world'])
    state=dict(completed_steps=row['completed_steps'],source_world=row['source_world'],
        source_training=str(target),consumed_global_samples=row['consumed_global_samples'],
        metadata_sha256=row['metadata_sha256'],checkpoint_file_sizes=row['checkpoint_file_sizes'],
        training_seconds_at_resume=row.get('training_seconds_at_resume',0.),updated_at_utc=now())
    save(work/'continuation_state.json',state)
    return state


def recover(work):
    initial=load(work/'resume_plan.json')
    state=load(work/'continuation_state.json') or dict(completed_steps=initial['completed_steps'],
        source_world=initial['source_world'],source_training=str(work/'source_snapshot'),
        consumed_global_samples=initial['consumed_global_samples'],training_seconds_at_resume=0.)
    attempts=[];candidates=[]
    for path in (work/'attempts').glob('*.json'):
        attempt=int(path.stem);attempts.append(attempt)
        plan=load(path);folder=work/plan['arm']/'training'
        for ready in (folder/'checkpoint_ready').glob('step_*.json'):
            row=load(ready)
            if row.get('completed_steps',0)>state['completed_steps']:
                candidates.append((row['completed_steps'],attempt,folder,row))
    for _,attempt,folder,row in sorted(candidates,key=lambda c:(c[0],c[1]),reverse=True):
        if valid_checkpoint(folder,row):
            state=pin(work,folder,row,attempt);break
    # A committed cursor must still point to its preserved checkpoint.
    source=Path(state['source_training'])
    if not (source/'ckpt'/str(state['completed_steps']-1)/'.metadata').is_file():
        raise RuntimeError('preserved_recovery_checkpoint_missing')
    if 'metadata_sha256' in state and not valid_checkpoint(source,state):
        raise RuntimeError('preserved_recovery_checkpoint_integrity_failed')
    return state,max(attempts,default=-1)+1
