"""Verify final synthetic checkpoints, then retain only that synthetic checkpoint.

Never touches real-training checkpoints or any earlier experiment.
"""
import json
import shutil
from pathlib import Path
import torch.distributed.checkpoint as dcp
from configuration import WORK, ARMS, WORLD, CAPACITY_STEPS
from common import load, save, now, digest


def main():
    proof_path=WORK/'capacity_storage_integrity.json'
    if proof_path.exists():
        print('Synthetic checkpoint storage already verified.')
        return
    proofs=[]
    for arm in ARMS:
        folder=WORK/arm/'capacity'
        completion=load(folder/'completion.json')
        if completion.get('state')!='capacity_passed' or completion.get('completed_steps')!=CAPACITY_STEPS:
            print('Capacity checks still in progress; no files changed.')
            return
        receipts=[load(folder/f'rank_{i:02d}_receipt.json') for i in range(WORLD)]
        if not all(r.get('success') and r.get('completed_steps')==CAPACITY_STEPS for r in receipts):
            raise RuntimeError('capacity_receipts_incomplete')
        final=folder/'ckpt'/str(CAPACITY_STEPS-1)
        metadata=dcp.FileSystemReader(final).read_metadata()
        keys=metadata.state_dict_metadata.keys()
        if not all(any(k.startswith(prefix) for k in keys) for prefix in ('model.student.', 'model.teacher.', 'optimizer.state.')):
            raise RuntimeError('required_checkpoint_state_missing')
        if 'iteration' not in keys:raise RuntimeError('checkpoint_iteration_missing')
        files=set()
        for info in metadata.storage_data.values():
            path=final/info.relative_path
            if path.parent!=final or not path.is_file() or path.stat().st_size < info.offset+info.length:
                raise RuntimeError('checkpoint_file_missing_or_truncated')
            files.add(path)
        proofs.append(dict(arm=arm,completed_steps=CAPACITY_STEPS,model_and_optimizer_present=True,
            all_storage_extents_present=True,metadata_sha256=digest(final/'.metadata'),
            file_count=len(files),bytes=sum(p.stat().st_size for p in files)))
    removed=0
    for arm in ARMS:
        ckpts=WORK/arm/'capacity/ckpt'
        for path in ckpts.iterdir():
            if path.is_dir() and path.name.isdigit() and int(path.name)<CAPACITY_STEPS-1:
                shutil.rmtree(path)
                removed+=1
    save(proof_path,dict(state='passed',arms=proofs,earlier_synthetic_checkpoints_removed=removed,
        final_synthetic_checkpoints_retained=True,real_training_checkpoints_touched=False,created_at_utc=now()))
    print('Both final synthetic checkpoints verified; earlier temporary synthetic checkpoints removed; real-training checkpoints untouched.')


if __name__=='__main__':main()
