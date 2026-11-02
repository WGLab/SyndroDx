"""Unique cached-feature diagnostic tasks shared by early-arriving GPU leases."""
import contextlib
import fcntl
import time
from configuration import WORK
from common import load,save


@contextlib.contextmanager
def locked_ledger():
    with (WORK/'staging/ledger.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        ledger=load(WORK/'staging/ledger.json')
        yield ledger
        save(WORK/'staging/ledger.json',ledger)


def claim(owner,preferred=None):
    with locked_ledger() as ledger:
        ready=[key for key,value in ledger['tasks'].items() if value['state']=='pending']
        ready.sort(key=lambda key:(not key.startswith(str(preferred)+'_'),key))
        if not ready:return None
        key=ready[0];value=ledger['tasks'][key]
        value.update(state='claimed',owner=owner,claimed_at=time.time())
        return key,dict(value)


def finish(key,owner,*,complete):
    with locked_ledger() as ledger:
        value=ledger['tasks'][key]
        if value.get('owner')!=owner:raise RuntimeError('diagnostic_task_owner_mismatch')
        value['state']='complete' if complete else 'pending'
        value.pop('owner',None)


def prepare():
    import json
    import numpy as np
    from validation_hook import cohort
    folder=WORK/'staging';folder.mkdir(exist_ok=True)
    records,_,_=cohort()
    with (folder/'inputs.private.jsonl').open('w') as stream:
        for record in records:stream.write(json.dumps(record)+'\n')
    old=WORK.with_name('cranio_only_cpt_20261021_single16')/'cranio_single16/training'
    baseline=np.load(WORK/'original_runtime_baseline.private.npy',allow_pickle=False)
    cases={'0':dict(checkpoint=None,reference=str(WORK/'original_runtime_baseline.private.npy'))}
    for step in (500,525,550,575):
        checkpoint=old/'ckpt'/str(step-1)
        if not (checkpoint/'.metadata').exists():raise RuntimeError('staging_checkpoint_missing')
        value=np.empty_like(baseline)
        for rank in range(16):
            shard=np.load(old/'validation'/f'step_{step:06d}'/f'rank_{rank:02d}.private.npy',allow_pickle=False)
            if shard.shape!=value[rank::16].shape:raise RuntimeError('staging_reference_shape_mismatch')
            value[rank::16]=shard
        reference=folder/f'checkpoint_{step}.reference.private.npy'
        np.save(reference,value)
        cases[str(step)]=dict(checkpoint=str(checkpoint),reference=str(reference))
    tasks={}
    for case in cases:
        for start in range(0,len(records),128):
            tasks[f'{int(case):04d}_{start:06d}']=dict(state='pending',case=case,start=start,stop=min(start+128,len(records)))
    save(folder/'manifest.json',dict(cases=cases,rows=len(records),pooling='L2 normalized mean of patch tokens',
        existing_CLS_features_not_reextracted_as_an_endpoint=True,patients_remain_local=True))
    save(folder/'ledger.json',dict(tasks=tasks))


if __name__=='__main__':prepare()
