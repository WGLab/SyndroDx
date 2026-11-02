"""Measure complete ArcFace optimizer steps on synthetic inputs only."""
import gc
import time
import torch
import torch.nn.functional as F
from control import WORK,save,now,load
from model import Encoder
from train import make_optimizer

def main():
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    p=load(WORK/'protocol.json')
    student=Encoder(p['num_classes'],p['arcface_scale'],p['arcface_margin']).cuda().train()
    optimizer=make_optimizer(student,p);total=torch.cuda.get_device_properties(0).total_memory
    measurements=[];batch=256;oom=False
    for trial in range(5):
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        try:
            for iteration in range(2):
                images=torch.randn(batch,3,112,112,device='cuda')
                labels=torch.randint(p['num_classes'],(batch,),device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=student(images,labels)
                loss=F.cross_entropy(output.float(),labels)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite_synthetic_loss')
                loss.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all() for v in student.parameters() if v.requires_grad):
                    raise RuntimeError('gpu_gradient_coverage_failure')
                torch.nn.utils.clip_grad_norm_(student.parameters(),3.,error_if_nonfinite=True)
                optimizer.step()
                del images,labels,output,loss
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            oom=True;break
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        measurements.append(dict(batch=batch,peak_gib=peak/2**30,reserved_gib=reserved/2**30,
            seconds_per_step=(time.monotonic()-started)/2))
        save(WORK/'capacity_progress.json',dict(state='measuring',measurements=measurements,updated_at_utc=now()))
        target=int(batch*(.72*total)/max(peak,1)//16)*16
        next_batch=min(2048,max(batch,min(batch*2,target)))
        if next_batch==batch:break
        batch=next_batch
    admitted=[m for m in measurements if m['peak_gib']*2**30<.80*total]
    if not admitted:raise RuntimeError('no_safe_gpu_batch')
    save(WORK/'capacity_receipt.json',dict(state='passed',device=torch.cuda.get_device_name(),total_gib=total/2**30,
        selected_per_gpu_batch=max(admitted,key=lambda x:x['batch'])['batch'],synthetic_only=True,
        oom_boundary_encountered=oom,measurements=measurements,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'capacity_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise
