"""Pure policies and local immutable snapshots for checkpoint-boundary resizing."""
import os
from pathlib import Path
from common import load,save,digest,now

SUPPORTED_WORLDS=(4,5,8,10,16,20)


def choose_world(available,consumed,current=0,minimum=4):
    eligible=[n for n in SUPPORTED_WORLDS if minimum<=n<=available and n>current and consumed%n==0]
    return max(eligible,default=current)


def exposure(consumed,resume_step,completed_step,world,batch=32):
    if completed_step<resume_step:raise RuntimeError('checkpoint_step_regressed')
    return consumed+(completed_step-resume_step)*world*batch


def verified_boundary(folder,world):
    rows=[load(folder/f'rank_{r:02d}_receipt.json') for r in range(world)]
    if not all(r.get('success') for r in rows) or len({(r.get('state'),r.get('completed_steps'),r.get('consumed_global_samples')) for r in rows})!=1:
        raise RuntimeError('inconsistent_pool_completion_receipts')
    row=rows[0];step=row['completed_steps']
    if row['state']=='paused_for_pool_expansion':
        report=load(folder/'validation'/f'step_{step:06d}'/'report.json')
        if not report.get('decision',{}).get('continue_training'):
            raise RuntimeError('resize_requires_passing_validation')
        if step%25 or not (folder/'ckpt'/str(step-1)/'.metadata').is_file():
            raise RuntimeError('resize_requires_complete_checkpoint')
        if not all(load(folder/f'resume_rank_{r:02d}.json').get('state')=='passed' for r in range(world)):
            raise RuntimeError('resize_requires_verified_source_restore')
    return row


def snapshot(folder,target,step,world):
    """Hardlinks preserve a fully written DCP checkpoint after all ranks exit."""
    if target.exists():raise RuntimeError('snapshot_must_be_new')
    checkpoint=folder/'ckpt'/str(step-1)
    if not (checkpoint/'.metadata').is_file():raise RuntimeError('checkpoint_metadata_required')
    target.mkdir(parents=True)
    paths=list(checkpoint.iterdir())+[folder/'config.yaml']
    paths+=list((folder/'validation'/f'step_{step:06d}').glob('*'))
    paths += [folder/'validation'/name for name in ('last_accepted.json','monitor_state.json')]
    for source in paths:
        if not source.is_file():continue
        destination=target/source.relative_to(folder)
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,destination)
    for rank in range(world):
        if not (target/'validation'/f'step_{step:06d}'/f'rank_{rank:02d}.private.npy').is_file():
            raise RuntimeError('snapshot_validation_shards_incomplete')
    save(target/'snapshot_receipt.json',dict(state='passed',completed_steps=step,source_world=world,
        metadata_sha256=digest(checkpoint/'.metadata'),checkpoint_file_sizes={p.name:p.stat().st_size for p in checkpoint.iterdir() if p.is_file()},
        created_at_utc=now()))
    return target
