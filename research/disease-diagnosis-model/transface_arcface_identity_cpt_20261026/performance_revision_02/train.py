"""Distributed patient-identity ArcFace adaptation with complete checkpoints."""
import datetime
import json
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from control import WORK,load,save,now,digest,notify
from model import Encoder
from data import ClinicalFaces,StreamSampler

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

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
            if params:groups.append(dict(params=params,lr=lr,base_lr=lr,weight_decay=wd if decay else 0.,family=family))
    return torch.optim.AdamW(groups,betas=(.9,.999))

def checkpoint(student,optimizer,step,consumed,seconds,protocol_sha):
    rank,world=dist.get_rank(),dist.get_world_size()
    states=[None]*world if rank==0 else None
    dist.gather_object(rng_state(),states,dst=0)
    if rank==0:
        path=WORK/'checkpoints'/f'step_{step:07d}.pt';tmp=path.with_suffix('.tmp')
        state=dict(student=student.state_dict(),optimizer=optimizer.state_dict(),step=step,
            consumed_samples=consumed,training_seconds=seconds,rng_states=states,world_size=world,
            protocol_sha256=protocol_sha,class_map_sha256=load(WORK/'protocol.json')['class_map_sha256'],
            objective='patient_identity_arcface')
        with tmp.open('wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        tmp.replace(path)
        receipt=dict(state='complete',path=str(path),sha256=digest(path),step=step,consumed_samples=consumed,
            world_size=world,training_seconds=seconds,objective='patient_identity_arcface',updated_at_utc=now())
        save(path.with_suffix('.json'),receipt);save(WORK/'latest_checkpoint.json',receipt)
        from retention import rotate
        rotate()
    dist.barrier()

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
    if digest(WORK/'model.py')!=load(WORK/'code_receipt.json')['model.py']:
        raise RuntimeError('model_code_modified_after_preflight')
    if digest(WORK/p['class_map'])!=p['class_map_sha256']:
        raise RuntimeError('identity_class_map_modified')
    student=Encoder(p['num_classes'],p['arcface_scale'],p['arcface_margin']).cuda()
    optimizer=make_optimizer(student,p)
    resume=load(WORK/'latest_checkpoint.json');step=0;consumed=0;seconds=0.
    if resume:
        if digest(resume['path'])!=resume['sha256']:raise RuntimeError('checkpoint_digest_mismatch')
        state=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=sha or state['class_map_sha256']!=p['class_map_sha256']:
            raise RuntimeError('checkpoint_protocol_or_identity_map_mismatch')
        student.load_state_dict(state['student'],strict=True);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];consumed=state['consumed_samples'];seconds=state['training_seconds']
        if state['world_size']==world:restore_rng(state['rng_states'][rank])
        else:
            seed=20261026+consumed+rank
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
        del state
    student.train();wrapped=DDP(student,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    if not resume:
        random.seed(20261026+rank);np.random.seed(20261026+rank);torch.manual_seed(20261026+rank)
    if rank==0:save(WORK/'status.json',dict(state='loaded',completed_steps=step,world_size=world,
        per_gpu_batch=batch,global_batch=batch*world,attempt=attempt,updated_at_utc=now()))
    from evaluate import validate
    if not resume:
        if rank==0:save(WORK/'status.json',dict(state='baseline_validation',completed_steps=0,world_size=world,updated_at_utc=now()))
        validate(student,0,attempt)
        checkpoint(student,optimizer,0,0,0.,sha)
        if rank==0:
            from retention import record_validation
            record_validation(0)
        dist.barrier()
    dataset=ClinicalFaces()
    loader=DataLoader(dataset,batch_size=batch,sampler=StreamSampler(consumed,rank,world,batch),
        num_workers=3,pin_memory=True,persistent_workers=True,prefetch_factor=3,drop_last=True)
    iterator=iter(loader)
    while step<1000000:
        student.train()
        started=time.monotonic()
        images,labels=next(iterator);data_wait=time.monotonic()-started
        images=images.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
        set_learning_rates(optimizer,step,p);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=wrapped(images,labels)
        loss=F.cross_entropy(output.float(),labels)
        good=torch.isfinite(loss).to(torch.int);dist.all_reduce(good,op=dist.ReduceOp.MIN)
        if not bool(good):
            if rank==0:save(WORK/'numerical_failure.json',dict(reason='nonfinite_arcface_loss',completed_steps=step,attempt=attempt,updated_at_utc=now()))
            raise FloatingPointError('nonfinite_arcface_loss')
        loss.backward()
        if step<p['head_only_steps']:
            for param in student.backbone.parameters():param.grad=None
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),p['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize();step+=1;consumed+=batch*world;elapsed=time.monotonic()-started;seconds+=elapsed
        measures=torch.stack((loss.detach(),(output.detach().argmax(1)==labels).float().mean()))
        dist.all_reduce(measures);measures/=world
        if rank==0 and (step<=5 or step%10==0):
            row=dict(state='training',objective='patient_identity_arcface',completed_steps=step,
                stage='identity_head_warmup' if step<=p['head_only_steps'] else 'identity_backbone_cpt',
                loss=float(measures[0]),training_margin_top1=float(measures[1]),gradient_norm=float(norm),
                world_size=world,per_gpu_batch=batch,global_batch=batch*world,
                backbone_lr=optimizer.param_groups[0]['lr'],step_seconds=elapsed,images_per_second=batch*world/elapsed,
                consumed_samples=consumed,active_training_seconds=seconds,attempt=attempt,data_wait_seconds=data_wait,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,updated_at_utc=now())
            save(WORK/'status.json',row)
            with (WORK/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step%p['checkpoint_every_steps']==0:
            if rank==0:save(WORK/'status.json',dict(state='checkpointing',completed_steps=step,world_size=world,updated_at_utc=now()))
            checkpoint(student,optimizer,step,consumed,seconds,sha)
            if step in p['validation_first_steps'] or step%p['validation_every_steps']==0:
                state=rng_state()
                if rank==0:save(WORK/'status.json',dict(state='validation',completed_steps=step,world_size=world,updated_at_utc=now()))
                validate(student,step,attempt);restore_rng(state)
                if rank==0:
                    from retention import record_validation
                    record_validation(step)
                dist.barrier()
                p['runtime_backbone_lr_multiplier']=load(WORK/'adaptive_lr.json').get('multiplier',1.)
                if rank==0 and step in (100,250):
                    notify('validation '+str(step),'Patient-identity ArcFace development validation completed',
                        'A scheduled internal development check completed. Aggregate metrics and protected checkpoints remain on HPC. No patient images or identifiers are included.')
            request=load(WORK/'resize_requested.json')
            stop=torch.tensor(int(request.get('attempt')==attempt or (WORK/'stop_requested.json').exists()),device='cuda')
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if bool(stop):
                if rank==0:save(WORK/'attempt_exit.json',dict(state='checkpointed_resize_or_stop',attempt=attempt,step=step,updated_at_utc=now()))
                break
        del loss,output,images,labels
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    try:train()
    except Exception as error:
        rank=int(os.environ.get('RANK','0'))
        save(WORK/'failures'/f"attempt_{os.environ.get('ARC_ATTEMPT','unknown')}_rank_{rank}.json",
            dict(state='failed',error_type=type(error).__name__,oom=isinstance(error,torch.cuda.OutOfMemoryError),updated_at_utc=now()))
        raise
