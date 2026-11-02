"""Synthetic-only sixteen-GPU collective and 8-to-16 full-state resume test."""
import datetime
import os
import time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as sd
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from configuration import WORK, WORLD
from common import save, now
from dinov3.checkpointer import load_checkpoint
from continuation import restore_full_state


def model(mesh):
    torch.manual_seed(123)
    result=torch.nn.ModuleDict(dict(student=torch.nn.Linear(128,128,device='cuda'),
                                    teacher=torch.nn.Linear(128,128,device='cuda')))
    result.teacher.load_state_dict(result.student.state_dict())
    result.teacher.requires_grad_(False)
    fully_shard(result.student,mesh=mesh)
    fully_shard(result.teacher,mesh=mesh)
    return result


def update(value,optimizer):
    x=torch.linspace(-1.,1.,512,device='cuda').reshape(4,128)
    optimizer.zero_grad()
    value.student(x).square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        for teacher,student in zip(value.teacher.parameters(),value.student.parameters()):
            teacher.mul_(.9).add_(student,alpha=.1)


def flatten(value):
    tensors=[]
    for key,tensor in sorted(sd.get_model_state_dict(value).items()):
        if hasattr(tensor,'full_tensor'):tensor=tensor.full_tensor()
        tensors.append(tensor.detach().reshape(-1))
    return torch.cat(tensors)


def main():
    rank=int(os.environ['RANK'])
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    if dist.get_world_size()!=WORLD:raise RuntimeError('probe_world_size_mismatch')
    # Four rounds cross both node and allocation boundaries. No model/data input.
    scalar=torch.tensor(float(rank+1),device='cuda')
    dist.all_reduce(scalar)
    if float(scalar)!=WORLD*(WORLD+1)/2:raise RuntimeError('collective_sum_mismatch')
    payload=torch.ones(16*1024*1024,device='cuda')
    torch.cuda.synchronize();started=time.monotonic()
    for _ in range(4):
        payload.fill_(1.)
        dist.all_reduce(payload)
    torch.cuda.synchronize();seconds=(time.monotonic()-started)/4
    if not bool((payload==WORLD).all()):raise RuntimeError('collective_payload_mismatch')
    del payload
    group=dist.new_group(list(range(8)))
    before=after=None
    if rank<8:
        mesh=DeviceMesh.from_group(group,'cuda')
        source=model(mesh)
        optimizer=torch.optim.AdamW(source.student.parameters(),lr=.003)
        for _ in range(3):update(source,optimizer)
        before=flatten(source)
        dcp.save(dict(iteration=2,model=sd.get_model_state_dict(source),
            optimizer=sd.get_optimizer_state_dict(source,optimizer)),
            checkpoint_id=WORK/'gpu_probe'/'synthetic_checkpoint8',process_group=group)
        update(source,optimizer)
        after=flatten(source)
        del source,optimizer
    else:
        before=torch.empty(2*(128*128+128),device='cuda')
        after=torch.empty_like(before)
    dist.barrier()
    dist.broadcast(before,src=0);dist.broadcast(after,src=0)
    target=model(init_device_mesh('cuda',(WORLD,)))
    optimizer=torch.optim.AdamW(target.student.parameters(),lr=.003)
    _,proof=restore_full_state(WORK/'gpu_probe'/'synthetic_checkpoint8',model=target,
        optimizer=optimizer,loader=load_checkpoint,completed_steps=3)
    if not torch.allclose(flatten(target),before,rtol=0,atol=1e-7):
        raise RuntimeError('synthetic_model_or_teacher_reshard_mismatch')
    update(target,optimizer)
    error=float((flatten(target)-after).abs().max())
    if error>1e-7:raise RuntimeError('synthetic_optimizer_next_update_mismatch')
    save(WORK/'gpu_probe'/f'rank_{rank:02d}.json',dict(state='passed',world_size=WORLD,
        rank=rank,allreduce_seconds=seconds,optimizer_reshard_next_update_matches=True,
        maximum_next_update_difference=error,**proof,completed_at_utc=now()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
