"""Elastic-at-checkpoint distributed alignment; only aggregate metrics leave workers."""
import datetime
import math
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from common import *
from model import Alignment,semantic_loss,targets

def gather(x):return torch.cat(dist_nn.all_gather(x),dim=0)

def macro_top1(pred,truth):
    return float(np.mean([np.mean(pred[truth==c]==truth[truth==c]) for c in np.unique(truth)]))

@torch.no_grad()
def evaluate(model,features,patients,arrays,train_ids,val_ids):
    model.eval();images=[];texts=[]
    for start in range(0,len(val_ids),64):
        ids=val_ids[start:start+64]
        x=torch.stack([features[patients[int(i)]['feature_indices']].mean(0) for i in ids])
        x=F.normalize(x,dim=-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            im,txt,scale=model(x,arrays['input_ids'][ids],arrays['attention_mask'][ids])
        images.append(im.float());texts.append(txt.float())
    im=torch.cat(images);txt=torch.cat(texts)
    hp=arrays['hpo'][val_ids];labels=arrays['labels'][val_ids]
    loss=semantic_loss(im,txt,scale,hp,labels)
    positive,_=targets(hp,labels);order=(im@txt.T).argmax(1)
    semantic_hit=float((positive[torch.arange(len(val_ids),device='cuda'),order]>0).float().mean())
    tr=torch.stack([features[patients[int(i)]['feature_indices']].mean(0) for i in train_ids])
    tr=F.normalize(model.image_projection(F.normalize(tr,dim=-1)),dim=-1)
    nearest=(im@tr.T).argmax(1)
    predicted=arrays['labels'][train_ids][nearest].cpu().numpy();truth=labels.cpu().numpy()
    result=dict(loss=float(loss),image_to_HPO_semantic_top1=semantic_hit,
        disease_retrieval_macro_top1=macro_top1(predicted,truth),validation_patients=len(val_ids),
        image_embedding_effective_rank=float(torch.exp(-(lambda s:(s/s.sum())*(s/s.sum()).clamp_min(1e-12).log())(torch.linalg.svdvals(im-im.mean(0))).sum())))
    model.train();return result

def checkpoint(model,optimizer,step,seen,best,bad,epoch,protocol_hash,world):
    target=WORK/'checkpoints'/f'step_{step:07d}.pt';tmp=target.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seen=seen,
            best=best,bad=bad,epoch=epoch,protocol_sha256=protocol_hash,world=world),f)
        f.flush();os.fsync(f.fileno())
    tmp.replace(target)
    save(WORK/'latest_checkpoint.json',dict(state='complete',path=str(target),sha256=digest(target),
        step=step,seen=seen,epoch=epoch,world=world,updated_at_utc=now()))
    return target

def main():
    os.umask(0o077);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    attempt=int(os.environ['ALIGN_ATTEMPT']);torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=8))
    torch.manual_seed(20261027);np.random.seed(20261027)
    receipt=load(WORK/'feature_receipt.json')
    if receipt.get('state')!='complete':raise RuntimeError('sealed_features_required')
    for path,sha in receipt['files'].items():
        if digest(WORK/path)!=sha:raise RuntimeError('training_input_modified')
    p=load(WORK/'protocol.json');protocol_hash=digest(WORK/'protocol.json')
    patients=load(WORK/'private/training_index.json')['patients']
    features=torch.from_numpy(np.load(WORK/'private/image_features.npy')).cuda()
    with np.load(WORK/'private/text_arrays.npz') as data:
        arrays={k:torch.from_numpy(np.array(data[k])).cuda() for k in data.files}
    train_ids=np.asarray([i for i,r in enumerate(patients) if r['split']=='train'])
    val_ids=np.asarray([i for i,r in enumerate(patients) if r['split']=='validation'])
    batch=min(int(os.environ.get('ALIGN_BATCH','96')),max(1,min(p['target_global_batch'],len(train_ids))//world))
    global_batch=batch*world
    if global_batch>len(train_ids):raise RuntimeError('batch_exceeds_unique_training_patients')
    model=Alignment().cuda()
    opt=torch.optim.AdamW([
        dict(params=[v for v in model.text.parameters() if v.requires_grad],lr=p['text_lr'],base_lr=p['text_lr']),
        dict(params=list(model.image_projection.parameters())+list(model.text_projection.parameters())+[model.logit_scale],lr=p['projection_lr'],base_lr=p['projection_lr'])],weight_decay=.01)
    step=0;seen=0;epoch=0;best=math.inf;bad=0
    resume=load(WORK/'latest_checkpoint.json')
    if resume:
        if digest(resume['path'])!=resume['sha256']:raise RuntimeError('alignment_checkpoint_integrity_failure')
        s=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if s['protocol_sha256']!=protocol_hash:raise RuntimeError('resume_protocol_changed')
        model.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer'])
        step,seen,epoch,best,bad=[s[k] for k in ['step','seen','epoch','best','bad']];del s
    wrapped=DDP(model,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    if rank==0:
        save(WORK/'status.json',dict(state='training',phase='image_HPO_alignment',completed_steps=step,
            world_size=world,per_gpu_batch=batch,global_batch=global_batch,epoch=epoch,attempt=attempt,updated_at_utc=now()))
        if not resume:
            initial=evaluate(model,features,patients,arrays,train_ids,val_ids)
            save(WORK/'baseline_alignment.json',dict(initial,updated_at_utc=now()))
            checkpoint(model,opt,0,0,best,bad,0,protocol_hash,world)
    dist.barrier();model.train();last=time.monotonic();last_seen=seen
    next_boundary=(epoch+1)*len(train_ids)
    while epoch<p['max_epochs']:
        rng=np.random.default_rng(p['seed']+step)
        chosen=rng.choice(train_ids,size=global_batch,replace=False)
        local=chosen[rank*batch:(rank+1)*batch]
        picks=[patients[int(i)]['feature_indices'][int(rng.integers(len(patients[int(i)]['feature_indices'])))] for i in local]
        torch.manual_seed(p['seed']+step*world+rank)
        warm=min(1.,(step+1)/p['warmup_steps'])
        for group in opt.param_groups:group['lr']=group['base_lr']*warm
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            image,text,scale=wrapped(features[picks],arrays['input_ids'][local],arrays['attention_mask'][local])
        images=gather(image);texts=gather(text)
        loss=semantic_loss(images,texts,scale,arrays['hpo'][chosen],arrays['labels'][chosen])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite_alignment_loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite_alignment_gradient')
        opt.step();step+=1;seen+=global_batch
        if rank==0:
            item=dict(state='training',phase='image_HPO_alignment',completed_steps=step,loss=float(loss.detach()),
                world_size=world,per_gpu_batch=batch,global_batch=global_batch,epoch_fraction=seen/len(train_ids),
                training_patients=len(train_ids),validation_patients=len(val_ids),seen=seen,
                pairs_per_second=(seen-last_seen)/max(.001,time.monotonic()-last),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,attempt=attempt,updated_at_utc=now())
            save(WORK/'status.json',item)
            with (WORK/'metrics.jsonl').open('a') as f:f.write(__import__('json').dumps(item)+'\n')
        if seen<next_boundary:continue
        epoch+=1;next_boundary=(epoch+1)*len(train_ids)
        decision=torch.zeros(1,dtype=torch.int64,device='cuda')
        if rank==0:
            metrics=evaluate(model,features,patients,arrays,train_ids,val_ids)
            if not math.isfinite(metrics['loss']):raise RuntimeError('nonfinite_development_loss')
            improved=metrics['loss']<best-p['min_delta']
            if improved:best=metrics['loss'];bad=0
            else:bad+=1
            path=checkpoint(model,opt,step,seen,best,bad,epoch,protocol_hash,world)
            report=dict(metrics,epoch=epoch,step=step,best_validation_loss=best,nonimproving_epochs=bad,updated_at_utc=now())
            save(WORK/'validation'/f'epoch_{epoch:03d}.json',report);save(WORK/'latest_validation.json',report)
            if improved:
                best_path=WORK/'checkpoints/best.pt';tmp=best_path.with_suffix('.tmp');tmp.unlink(missing_ok=True);os.link(path,tmp);tmp.replace(best_path)
                save(WORK/'best_checkpoint.json',dict(path=str(best_path),sha256=digest(best_path),step=step,epoch=epoch,validation=report))
            stop=epoch>=p['max_epochs'] or (epoch>=p['min_epochs'] and bad>=p['patience'])
            resize=load(WORK/'resize_requested.json').get('world_size',world)!=world
            decision[0]=1 if stop else 2 if resize else 0
            if stop:
                save(WORK/'training_complete.json',dict(state='complete',reason='maximum_epochs' if epoch>=p['max_epochs'] else 'development_plateau',
                    completed_steps=step,epoch=epoch,best_checkpoint=load(WORK/'best_checkpoint.json'),updated_at_utc=now()))
        dist.broadcast(decision,src=0);dist.barrier()
        if decision.item():break
    dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'attempts'/f'{os.environ.get("ALIGN_ATTEMPT","0")}_rank_{os.environ.get("RANK","0")}.failure.json',
            dict(error_type=type(e).__name__,oom=isinstance(e,torch.cuda.OutOfMemoryError),updated_at_utc=now()))
        raise
