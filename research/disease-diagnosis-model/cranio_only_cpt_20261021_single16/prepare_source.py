"""Preserve the first repetition's latest fully validated immutable checkpoint."""
import hashlib
import json
import os
import shutil
from pathlib import Path

WORK=Path(__file__).resolve().parent
OLD=WORK.with_name('cranio_only_cpt_20261021_continue1000')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    source=OLD/'cranio_seed19'/'training'
    candidates=[]
    for path in (source/'ckpt').glob('*/.metadata'):
        if not path.parent.name.isdigit():continue
        step=int(path.parent.name)+1
        report=source/'validation'/f'step_{step:06d}'/'report.json'
        if not report.exists():continue
        value=json.loads(report.read_text())
        if value.get('state')=='complete':
            candidates.append((step,value))
    if not candidates:raise RuntimeError('no_complete_passing_source_checkpoint')
    step,report=max(candidates,key=lambda x:x[0])
    if not report['decision']['continue_training']:
        raise RuntimeError('latest_source_checkpoint_failed_guard_no_rollback_to_bypass_it')
    snapshot=WORK/'source_snapshot'
    snapshot.mkdir(mode=0o700)
    ckpt=snapshot/'ckpt'/str(step-1)
    # Checkpoint writes are atomic directory renames. Hard links preserve all
    # immutable source shards even if the previous controller's retention runs.
    shutil.copytree(source/'ckpt'/str(step-1),ckpt,copy_function=os.link)
    val=snapshot/'validation'/f'step_{step:06d}'
    shutil.copytree(source/'validation'/f'step_{step:06d}',val,copy_function=os.link)
    shutil.copy2(source/'config.yaml',snapshot/'config.yaml')
    history=snapshot/'validation'
    (history/'monitor_state.json').write_text(json.dumps(dict(completed_steps=step,
        supported_declines=report['decision'].get('supported_declines',[])),indent=2)+'\n')
    (history/'last_accepted.json').write_text(json.dumps(dict(completed_steps=step,
        metrics=report['metrics']),indent=2)+'\n')
    checked=[ckpt/'.metadata',snapshot/'config.yaml']+list(history.rglob('*.json'))+list(val.glob('*.npy'))
    proof=dict(state='source_preserved',completed_steps=step,source_world=8,target_world=16,
        source_arm='cranio_seed19',selection='fixed first repetition; latest complete guarded checkpoint',
        source_checkpoint=str(source/'ckpt'/str(step-1)),private_mapping_read=False,
        files={str(f.relative_to(snapshot)):digest(f) for f in checked},
        checkpoint_file_sizes={str(f.relative_to(snapshot)):f.stat().st_size for f in ckpt.iterdir() if f.is_file()})
    (WORK/'resume_plan.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(state='source_preserved',completed_steps=step,
        source_ranks=8,target_ranks=16,checkpoint_files=len(proof['checkpoint_file_sizes']))))


if __name__=='__main__':main()
