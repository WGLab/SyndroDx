"""Matched patient exposure across objectives, recoverable DDP on single-GPU leases."""
import os,sys,time,copy,math,random,datetime,contextlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from matched_runtime import *
from matched_models import Model,dino_loss,update_teacher
from matched_data import MatchedFaces
from data import StreamSampler
from extract import extract

def snapshot(student,teacher,opt,center,step,seconds,folder,p):
    rank=dist.get_rank();rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
    states=[None]*dist.get_world_size() if rank==0 else None
    dist.gather_object(rng,states,dst=0)
    if rank==0:
        path=folder/'checkpoints'/f'step_{step:07d}.pt';temp=path.with_suffix('.tmp')
        state=dict(student=student.state_dict(),teacher=teacher.state_dict() if teacher else None,optimizer=opt.state_dict(),
            center=center,step=step,training_seconds=seconds,rng=states,protocol_sha256=digest(ROOT/'protocol.json'))
        with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        temp.replace(path)
        receipt=dict(path=str(path),step=step,sha256=digest(path),protocol_sha256=state['protocol_sha256'],updated_at_utc=now())
        save(folder/'latest_checkpoint.json',receipt)
        # Keep two full recovery snapshots plus prespecified end points.
        paths=sorted((folder/'checkpoints').glob('step_*.pt'))
        for old in paths[:-2]:
            if int(old.stem.split('_')[1]) not in [1000,2000,p['max_steps']]:old.unlink()
    dist.barrier()

def rank0_decision(value):
    flag=torch.tensor([int(value) if dist.get_rank()==0 else 0],device='cuda',dtype=torch.int32)
    dist.broadcast(flag,src=0)
    return bool(flag.item())


def train():
    os.umask(0o077);arm=sys.argv[1];folder=ROOT/arm;p=load(ROOT/'protocol.json');a=p['arms'][arm]
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);attempt=int(os.environ['MATCH_ATTEMPT'])
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=True
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15))
    assert p['global_patient_draws_per_step']%world==0
    local=p['global_patient_draws_per_step']//world;maxmicro=int(os.environ.get('MATCH_MICRO','960'))
    seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    student=Model(a['objective'],p).cuda()
    teacher=Model(a['objective'],p).cuda().requires_grad_(False).eval() if a['objective'].startswith('dino') else None
    if teacher:teacher.load_state_dict(student.state_dict(),strict=True)
    groups=[]
    for name,module,lr,wd in [('backbone',student.backbone,a['backbone_lr'],.01),('head',student.head,a['head_lr'],.0 if teacher is None else .04)]:
        for decay in [True,False]:
            params=[v for n,v in module.named_parameters() if v.requires_grad and ((v.ndim>1 and not n.endswith('bias'))==decay)]
            groups.append(dict(params=params,lr=lr,base_lr=lr,family=name,weight_decay=wd if decay else 0.))
    opt=torch.optim.AdamW(groups,betas=(.9,.999));center=torch.zeros(1,p['dino_output_dim'],device='cuda')
    step=0;seconds=0.;resume=load(folder/'latest_checkpoint.json')
    if resume:
        assert digest(resume['path'])==resume['sha256']
        s=torch.load(resume['path'],map_location='cpu',weights_only=False)
        assert s['protocol_sha256']==digest(ROOT/'protocol.json')
        student.load_state_dict(s['student'],strict=True)
        if teacher:teacher.load_state_dict(s['teacher'],strict=True)
        opt.load_state_dict(s['optimizer']);center.copy_(s['center']);step=s['step'];seconds=s['training_seconds']
        if len(s['rng'])==world:
            r=s['rng'][rank];torch.set_rng_state(r['torch']);torch.cuda.set_rng_state(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
        else:torch.manual_seed(seed+rank+step)
        del s
    else:torch.manual_seed(seed+rank)
    student.train();wrapped=DDP(student,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    # DDP broadcasts student initialization; teacher must match it exactly.
    if teacher and not resume:teacher.load_state_dict(student.state_dict())
    if rank==0:save(folder/'status.json',dict(state='loading_data',step=step,world=world,attempt=attempt,updated_at_utc=now()))
    if not resume:
        if digest(GMDB/'private/images.uint8.npy')!=p['gmdb_image_array_sha256']:raise RuntimeError('gmdb_image_array_changed')
        extract(teacher or student,folder/'evaluations/step_0000000',0)
        snapshot(student,teacher,opt,center,0,0.,folder,p)
    ds=MatchedFaces()
    loader=DataLoader(ds,batch_size=local,sampler=StreamSampler(step*p['global_patient_draws_per_step'],rank,world,local),
        num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    it=iter(loader)
    while step<p['max_steps']:
        if rank0_decision((folder/'pause_requested.json').exists()):break
        started=time.monotonic();(x1,x2),labels=next(it);wait=time.monotonic()-started
        for g in opt.param_groups:
            factor=min(1.,max(0.,(step-p['head_only_steps']+1)/p['backbone_ramp_steps'])) if g['family']=='backbone' else min(1.,(step+1)/25.)
            g['lr']=g['base_lr']*factor
        opt.zero_grad(set_to_none=True);total=torch.zeros((),device='cuda');center_sum=torch.zeros_like(center)
        chunks=math.ceil(local/maxmicro);bounds=np.linspace(0,local,chunks+1,dtype=int)
        for c in range(chunks):
            lo,hi=int(bounds[c]),int(bounds[c+1]);n=hi-lo
            images=torch.cat([x1[lo:hi],x2[lo:hi]]).cuda(non_blocking=True);y=labels[lo:hi].cuda(non_blocking=True).repeat(2)
            sync=wrapped.no_sync() if c<chunks-1 else contextlib.nullcontext()
            with sync:
                if teacher:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):target=teacher(images)
                    center_sum.add_(target.float().sum(0,keepdim=True))
                with torch.autocast('cuda',dtype=torch.bfloat16):output=wrapped(images,y)
                if teacher:
                    t=p['teacher_temp_start']+(p['teacher_temp_end']-p['teacher_temp_start'])*min(1.,step/max(1,p['teacher_temp_warmup']-1))
                    loss=dino_loss(output,target,center,t)
                else:loss=F.cross_entropy(output.float(),y)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite_training_loss')
                (loss*n/local).backward();total.add_(loss.detach()*n/local)
            del images,output,loss,y
            if teacher:del target
        if step<p['head_only_steps']:
            for v in student.backbone.parameters():v.grad=None
        if teacher and step<10:
            for v in student.head.last_layer.parameters():v.grad=None
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),p['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        if teacher:
            dist.all_reduce(center_sum);center.mul_(p['center_momentum']).add_(center_sum,alpha=(1-p['center_momentum'])/(2*p['global_patient_draws_per_step']))
            update_teacher(student,teacher,p['teacher_momentum'])
        torch.cuda.synchronize();step+=1;elapsed=time.monotonic()-started;seconds+=elapsed
        dist.all_reduce(total);total/=world
        if step<=5 or step%10==0:
            telemetry=dict(rank=rank,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            try:telemetry['utilization_percent']=torch.cuda.utilization()
            except Exception:telemetry['utilization_percent']=None
            gathered=[None]*world if rank==0 else None;dist.gather_object(telemetry,gathered,dst=0)
            if rank==0:
                record=dict(state='training',arm=arm,step=step,loss=float(total),gradient_norm=float(norm),world=world,
                    global_patient_draws=p['global_patient_draws_per_step'],consumed_patient_draws=step*p['global_patient_draws_per_step'],
                    augmented_inputs=step*p['global_patient_draws_per_step']*2,step_seconds=elapsed,data_wait_seconds=wait,
                    training_seconds=seconds,backbone_lr=opt.param_groups[0]['lr'],gpu=gathered,attempt=attempt,updated_at_utc=now())
                save(folder/'status.json',record)
                with (folder/'metrics.jsonl').open('a') as f:f.write(__import__('json').dumps(record)+'\n')
        if step%p['checkpoint_every']==0 or step in p['validation_steps']:
            snapshot(student,teacher,opt,center,step,seconds,folder,p)
            if step in p['validation_steps']:
                extract(teacher or student,folder/'evaluations'/f'step_{step:07d}',step)
                if rank==0:
                    selected=teacher or student;path=folder/'evaluations'/f'step_{step:07d}'/'encoder.pt';temp=path.with_suffix('.tmp')
                    torch.save(dict(backbone=selected.backbone.state_dict(),step=step,protocol_sha256=digest(ROOT/'protocol.json')),temp);temp.replace(path)
            if rank0_decision((folder/'resize_requested.json').exists()):break
    if rank0_decision(bool(step and load(folder/'latest_checkpoint.json').get('step')!=step)):snapshot(student,teacher,opt,center,step,seconds,folder,p)
    if rank==0:save(folder/'exit.json',dict(state='complete' if step>=p['max_steps'] else 'checkpointed_pause_or_resize',step=step,attempt=attempt,updated_at_utc=now()))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    try:train()
    except Exception as e:
        folder=ROOT/sys.argv[1]
        save(folder/'failures'/f"attempt_{os.environ.get('MATCH_ATTEMPT')}_rank_{os.environ.get('RANK')}.json",dict(error_type=type(e).__name__,oom=isinstance(e,torch.cuda.OutOfMemoryError),updated_at_utc=now()))
        # Full exception text stays in HPC-private logs; tool output uses failure receipts.
        raise
