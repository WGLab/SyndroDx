"""Ask background seeds to checkpoint-yield; never cancel an allocation."""
from common import *


def main():
    definitions=[('fill','arcface_fill_seeds_20261028','fill_seeds.py'),
                 ('extra','arcface_extra_seeds_20261028','extra_seeds.py')]
    queue=[]
    # Queue may have advanced while this experiment was prepared. Resolve the
    # current round instead of reusing an earlier status snapshot.
    for kind,name,script in definitions:
        qflag=PROJECT/name/'stop_continuous_seeds.json'
        run=Path(load(PROJECT/name/'seed_queue_status.json')['path'])
        flags=[run/arm/'pause_requested.json' for arm in load(run/'protocol.json')['arms']
               if load(run/arm/'exit.json').get('state')!='complete']
        for path in [qflag,*flags]:
            if path.exists() and load(path).get('owner')!=str(WORK):raise ValueError('existing_pause_owned_elsewhere')
        queue.append(dict(kind=kind,queue=str(PROJECT/name),round=str(run),
                         queue_flag=str(qflag),arm_flags=list(map(str,flags)),
                         script=str(PROJECT/'arcface_transfer_evaluation_20261028'/script)))
    for old in load(WORK/'priority_yield.json').get('queues',[]):
        previous=Path(old.get('arm_flag','/nonexistent'))
        if load(previous).get('owner')==str(WORK):previous.unlink()
    for item in queue:
        for name in [item['queue_flag'],*item['arm_flags']]:
            save(name,dict(owner=str(WORK),reason='primary_matched_visual_unfreezing_trial_checkpoint_yield',
                                preserve_all_GPU_allocations=True,updated_at_utc=now()))
    save(WORK/'priority_yield.json',dict(queues=queue,requested_GPU_workers=9,parent_allocations_cancelled=0,updated_at_utc=now()))
    print('Checkpoint-boundary yield requested for two background seed groups; parent allocations retained.')


if __name__=='__main__':guarded(main,WORK/'priority_failure.json')
