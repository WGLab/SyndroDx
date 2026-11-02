"""Synthetic GPU capacity and trainability check before any clinical training."""
import gc
import os
import time
import torch
from control import WORK, save, now, load
from model import Encoder, update_teacher
from train import make_optimizer

def main():
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    p=load(WORK/'protocol.json')
    student=Encoder(p['output_dim']).cuda().train()
    teacher=Encoder(p['output_dim']).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(student.state_dict())
    optimizer=make_optimizer(student,p)
    total=torch.cuda.get_device_properties(0).total_memory
    measurements=[];batch=32
    for trial in range(6):
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        for iteration in range(3):
            images=torch.randn(2*batch,3,112,112,device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):target=teacher(images)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=student(images)
            q=torch.softmax(target.float()/.04,dim=-1).detach().chunk(2)
            logits=(output.float()/.1).chunk(2)
            loss=-(q[0]*torch.log_softmax(logits[1],dim=-1)).sum(-1).mean()/2
            loss-= (q[1]*torch.log_softmax(logits[0],dim=-1)).sum(-1).mean()/2
            loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),3.,error_if_nonfinite=True)
            optimizer.step();update_teacher(student,teacher,.996)
            del images,target,output,loss,q,logits
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        measurements.append(dict(batch=batch,peak_gib=peak/2**30,reserved_gib=reserved/2**30,
            seconds_per_step=(time.monotonic()-started)/3))
        save(WORK/'capacity_progress.json',dict(state='measuring',measurements=measurements,updated_at_utc=now()))
        if trial==5:break
        # Leave at least 20% for DDP buckets, validation and transient workspaces.
        target_batch=int(batch*(.72*total)/max(peak,1)//16)*16
        next_batch=min(768,max(batch,min(batch*2,target_batch)))
        if next_batch==batch:break
        batch=next_batch
    # Only choose a batch actually measured below the safety budget.
    admitted=[m for m in measurements if m['peak_gib']*2**30 < .80*total]
    if not admitted:raise RuntimeError('no_safe_gpu_batch')
    selected=max(admitted,key=lambda x:x['batch'])['batch']
    save(WORK/'capacity_receipt.json',dict(state='passed',device=torch.cuda.get_device_name(),
        total_gib=total/2**30,selected_per_gpu_batch=selected,synthetic_only=True,
        measurements=measurements,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'capacity_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
