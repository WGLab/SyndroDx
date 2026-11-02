"""GPU capacity and BF16 gradient check using synthetic data only."""
import time
import torch
from common import *
from model import Stage1

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.manual_seed(11)
    p=load(WORK/'protocol.json');b=p['batch_size'];m=Stage1().cuda().train()
    tokens=torch.randn(b,576,768,device='cuda',dtype=torch.float16);mask=torch.ones(b,576,device='cuda',dtype=torch.long)
    ids=torch.randint(1000,5000,(b,128),device='cuda');ids[:,0]=101;ids[:,-1]=102
    tm=torch.ones_like(ids);groups=torch.arange(b,device='cuda');present=torch.eye(b,dtype=torch.bool,device='cuda')
    opt=torch.optim.AdamW(m.parameters(),lr=p['learning_rate']);start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=m(tokens,mask,ids,tm,groups,present)
    assert torch.isfinite(loss);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
    save(WORK/'smoke.json',dict(state='passed',synthetic_only=True,batch_size=b,visual_tokens=576,text_tokens=128,
        parameters=sum(v.numel() for v in m.parameters()),three_losses={k:float(parts[k]) for k in ['itc','itm','itg']},
        gradient_norm=float(grad),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        step_seconds=time.monotonic()-start,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'smoke_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
