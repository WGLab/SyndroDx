"""One TransFace-L model across existing one-GPU allocations, with full resume."""
import datetime
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from control import WORK, load, save, now, digest, notify
from model import Encoder, update_teacher
from data import ClinicalFaces, StreamSampler
from vendor.dino_loss import DINOLoss

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

def checkpoint(student,teacher,optimizer,criterion,step,consumed,seconds,protocol_sha):
    rank,world=dist.get_rank(),dist.get_world_size()
    states=[None]*world if rank==0 else None
    dist.gather_object(rng_state(),states,dst=0)
    if rank==0:
        path=WORK/'checkpoints'/f'step_{step:07d}.pt'
        tmp=path.with_suffix('.tmp')
        state=dict(student=student.state_dict(),teacher=teacher.state_dict(),optimizer=optimizer.state_dict(),
            criterion=criterion.state_dict(),step=step,consumed_samples=consumed,training_seconds=seconds,
            rng_states=states,world_size=world,protocol_sha256=protocol_sha)
        with tmp.open('wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        tmp.replace(path)
        receipt=dict(state='complete',path=str(path),sha256=digest(path),step=step,
            consumed_samples=consumed,world_size=world,training_seconds=seconds,updated_at_utc=now())
        save(path.with_suffix('.json'),receipt);save(WORK/'latest_checkpoint.json',receipt)
        from retention import rotate
        rotate()
    dist.barrier()

def set_learning_rates(optimizer,step,protocol):
    warm=protocol['head_only_steps'];ramp=protocol['backbone_ramp_steps']
    factor=min(1.,max(0.,(step-warm+1)/ramp))
    for group in optimizer.param_groups:
        group['lr']=group['base_lr']*(factor*protocol.get('runtime_backbone_lr_multiplier',1.) if group['family']=='backbone' else min(1.,(step+1)/25))

def make_optimizer(student,p):
    groups=[]
    for family,module,lr,wd in [('backbone',student.backbone,p['backbone_learning_rate'],p['backbone_weight_decay']),
                                ('head',student.head,p['head_learning_rate'],p['head_weight_decay'])]:
        for decay in (True,False):
            params=[v for n,v in module.named_parameters() if v.requires_grad and ((v.ndim>1 and not n.endswith('bias'))==decay)]
            groups.append(dict(params=params,lr=lr,base_lr=lr,weight_decay=wd if decay else 0.,family=family))
    return torch.optim.AdamW(groups,betas=(.9,.999))

def train():
    os.umask(0o077)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);attempt=int(os.environ['ARC_ATTEMPT'])
    batch=int(os.environ['ARC_BATCH']);torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    random.seed(20261026);np.random.seed(20261026);torch.manual_seed(20261026)
    p=load(WORK/'protocol.json');sha=digest(WORK/'protocol.json')
    p['runtime_backbone_lr_multiplier']=load(WORK/'adaptive_lr.json').get('multiplier',1.)
    if digest(WORK/'model.py') != load(WORK/'code_receipt.json')['model.py']:
        raise RuntimeError('model_code_modified_after_preflight')
    student=Encoder(p['output_dim']).cuda();teacher=Encoder(p['output_dim']).cuda()
    teacher.load_state_dict(student.state_dict(),strict=True);teacher.requires_grad_(False);teacher.eval()
    optimizer=make_optimizer(student,p)
    # The unmodified official loss uses an epoch index; here its schedule index
    # is a completed optimizer step, with a declared 1000-step warm-up.
    criterion=DINOLoss(p['output_dim'],2,p['teacher_temperature_start'],p['teacher_temperature_end'],
        p['teacher_temperature_warmup_steps'],1000000,student_temp=p['student_temperature'],
        center_momentum=p['center_momentum']).cuda()
    resume=load(WORK/'latest_checkpoint.json');step=0;consumed=0;seconds=0.
    if resume:
        if digest(resume['path']) != resume['sha256']:
            raise RuntimeError('checkpoint_digest_mismatch')
        state=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if state['protocol_sha256'] != sha:
            raise RuntimeError('checkpoint_protocol_mismatch')
        student.load_state_dict(state['student'],strict=True);teacher.load_state_dict(state['teacher'],strict=True)
        optimizer.load_state_dict(state['optimizer']);criterion.load_state_dict(state['criterion'])
        step=state['step'];consumed=state['consumed_samples'];seconds=state['training_seconds']
        if state['world_size']==world:
            restore_rng(state['rng_states'][rank])
        else:
            seed=20261026+consumed+rank
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
        del state
    student.train();wrapped=DDP(student,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    if not resume:
        random.seed(20261026+rank);np.random.seed(20261026+rank);torch.manual_seed(20261026+rank)
    if rank==0:
        save(WORK/'status.json',dict(state='loaded',completed_steps=step,world_size=world,
            per_gpu_batch=batch,global_batch=batch*world,attempt=attempt,updated_at_utc=now()))
    from evaluate import validate
    if not resume:
        if rank==0:save(WORK/'status.json',dict(state='baseline_validation',completed_steps=0,world_size=world,updated_at_utc=now()))
        validate(teacher,0,attempt)
        checkpoint(student,teacher,optimizer,criterion,0,0,0.,sha)
        if rank==0:
            from retention import record_validation
            record_validation(0)
        dist.barrier()
    dataset=ClinicalFaces()
    loader=DataLoader(dataset,batch_size=batch,sampler=StreamSampler(consumed,rank,world,batch),
        num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,drop_last=True)
    iterator=iter(loader)
    origin=step
    while step<1000000:
        started=time.monotonic()
        first,second=next(iterator)
        data_wait=time.monotonic()-started
        images=torch.cat((first,second)).cuda(non_blocking=True)
        set_learning_rates(optimizer,step,p);optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            target=teacher(images)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=wrapped(images)
        loss=criterion(output.float(),target.float(),step)
        good=torch.isfinite(loss).to(torch.int);dist.all_reduce(good,op=dist.ReduceOp.MIN)
        if not bool(good):
            if rank==0:save(WORK/'numerical_failure.json',dict(reason='nonfinite_ssl_loss',completed_steps=step,attempt=attempt,updated_at_utc=now()))
            raise FloatingPointError('nonfinite_ssl_loss')
        loss.backward()
        if step<p['head_only_steps']:
            for param in student.backbone.parameters():param.grad=None
        if step<p['freeze_last_layer_steps']:
            for param in student.head.last_layer.parameters():param.grad=None
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),p['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step();update_teacher(student,teacher,p['teacher_momentum'])
        torch.cuda.synchronize();step+=1;consumed+=batch*world;elapsed=time.monotonic()-started;seconds+=elapsed
        mean=loss.detach();dist.all_reduce(mean);mean/=world
        if rank==0 and (step<=5 or step%10==0):
            row=dict(state='training',completed_steps=step,stage='head_warmup' if step<=p['head_only_steps'] else 'backbone_cpt',
                loss=float(mean),gradient_norm=float(norm),world_size=world,per_gpu_batch=batch,global_batch=batch*world,
                backbone_lr=optimizer.param_groups[0]['lr'],step_seconds=elapsed,images_per_second=batch*world/elapsed,
                consumed_samples=consumed,active_training_seconds=seconds,attempt=attempt,
                data_wait_seconds=data_wait,mean_training_images_per_second=consumed/max(seconds,1e-9),
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,updated_at_utc=now())
            save(WORK/'status.json',row)
            with (WORK/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step%p['checkpoint_every_steps']==0:
            if rank==0:save(WORK/'status.json',dict(state='checkpointing',completed_steps=step,world_size=world,updated_at_utc=now()))
            checkpoint(student,teacher,optimizer,criterion,step,consumed,seconds,sha)
            if step in p['validation_first_steps'] or step%p['validation_every_steps']==0:
                state=rng_state()
                if rank==0:save(WORK/'status.json',dict(state='validation',completed_steps=step,world_size=world,updated_at_utc=now()))
                safe=validate(teacher,step,attempt);restore_rng(state)
                if rank==0:
                    from retention import record_validation
                    record_validation(step)
                dist.barrier()
                p['runtime_backbone_lr_multiplier']=load(WORK/'adaptive_lr.json').get('multiplier',1.)
                if not safe:
                    if rank==0:notify('development guard','TransFace-L CPT checkpoint requires review',
                        'A development representation guard triggered. The checkpoint and GPU leases are preserved. Only aggregate reports are available on HPC.')
                    break
                if rank==0 and step in (100,250):
                    notify('validation '+str(step),'TransFace-L CPT development validation completed',
                        'The TransFace-L craniofacial self-supervised run completed a scheduled internal development check. Aggregate metrics and checkpoints remain on HPC; no images or identifiers are included.')
            request=load(WORK/'resize_requested.json')
            stop=torch.tensor(int(request.get('attempt')==attempt or (WORK/'stop_requested.json').exists()),device='cuda')
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if bool(stop):
                if rank==0:save(WORK/'attempt_exit.json',dict(state='checkpointed_resize_or_stop',attempt=attempt,step=step,updated_at_utc=now()))
                break
        del loss,target,output,images
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    try:
        train()
    except Exception as error:
        rank=int(os.environ.get('RANK','0'))
        save(WORK/'failures'/f"attempt_{os.environ.get('ARC_ATTEMPT','unknown')}_rank_{rank}.json",
            dict(state='failed',error_type=type(error).__name__,oom=isinstance(error,torch.cuda.OutOfMemoryError),updated_at_utc=now()))
        raise
