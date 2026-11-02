"""Matched patient-isolated image-only, HPO-only, and cross-attention probes."""
import datetime
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from common import *
from fusion_model import Fusion

@torch.no_grad()
def validate(model,data,ids):
    model.eval();scores=[]
    for start in range(0,len(ids),64):
        ix=ids[start:start+64]
        with torch.autocast('cuda',dtype=torch.bfloat16):s=model(data['features'][ix],data['input_ids'][ix],data['attention_mask'][ix])
        scores.append(s.float())
    scores=torch.cat(scores);truth=data['labels'][ids];order=scores.argsort(dim=1,descending=True)
    metrics={}
    for k in (1,5,10):
        hit=(order[:,:min(k,scores.shape[1])]==truth[:,None]).any(1).float()
        metrics['macro_top'+str(k)]=float(torch.stack([hit[truth==c].mean() for c in truth.unique()]).mean())
    metrics['loss']=float(F.cross_entropy(scores,truth));metrics['patients']=len(ids)
    model.train();return metrics

def main():
    os.umask(0o077);mode=sys.argv[1];folder=WORK/'fusion'/mode;folder.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(0);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=8));torch.manual_seed(20261028)
    receipt=load(WORK/'fusion_data_receipt.json')
    if digest(WORK/'private/fusion_arrays.npz')!=receipt['arrays_sha256']:raise RuntimeError('fusion_input_changed')
    with np.load(WORK/'private/fusion_arrays.npz') as d:data={k:torch.from_numpy(np.array(d[k])).cuda() for k in ['features','input_ids','attention_mask','labels']};train=d['train'].copy();val=d['validation'].copy()
    model=Fusion(receipt['classes'],mode).cuda()
    text=[p for p in model.alignment.text.parameters() if p.requires_grad]
    other=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('alignment.text.')]
    groups=[dict(params=other,lr=1e-4)]
    if text:groups.append(dict(params=text,lr=3e-6))
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    epoch=0;step=0;best=-1.;bad=0
    if (folder/'last.pt').exists():
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);epoch=s['epoch'];step=s['step'];best=s['best'];bad=s['bad']
    wrapped=DDP(model,device_ids=[0],broadcast_buffers=False)
    counts=torch.bincount(data['labels'][train],minlength=receipt['classes']).float();weights=counts.clamp_min(1).rsqrt();weights/=weights.mean()
    batch=min(64,max(1,min(384,len(train))//world));global_batch=batch*world
    model.train()
    for epoch in range(epoch+1,51):
        losses=[]
        for _ in range((len(train)+global_batch-1)//global_batch):
            rng=np.random.default_rng(20261028+step);chosen=rng.choice(train,size=global_batch,replace=False)[rank*batch:(rank+1)*batch]
            torch.manual_seed(20261028+step*world+rank)
            x=data['features'][chosen].clone();mask=data['attention_mask'][chosen].clone()
            if mode=='fusion':
                drop=torch.rand(len(chosen),device='cuda');x[drop<.1]=0
                mask[(drop>=.1)&(drop<.2),1:]=0
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=wrapped(x,data['input_ids'][chosen],mask);loss=F.cross_entropy(logits.float(),data['labels'][chosen],weight=weights)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite_diagnosis_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1;losses.append(float(loss.detach()))
            if rank==0:save(folder/'status.json',dict(state='training',mode=mode,epoch=epoch,step=step,loss=losses[-1],world_size=world,global_batch=global_batch,updated_at_utc=now()))
        decision=torch.zeros(1,dtype=torch.long,device='cuda')
        if rank==0:
            metrics=validate(model,data,val);score=metrics['macro_top5'];improved=score>best+1e-5
            best=max(best,score);bad=0 if improved else bad+1
            tmp=folder/'last.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,best=best,bad=bad),tmp);tmp.replace(folder/'last.pt')
            report=dict(metrics,epoch=epoch,step=step,training_loss=float(np.mean(losses)),world_size=world,updated_at_utc=now())
            save(folder/'latest_validation.json',report)
            if improved:
                target=folder/'best.pt';target.unlink(missing_ok=True);os.link(folder/'last.pt',target);save(folder/'best_validation.json',report)
            if epoch>=50 or (epoch>=15 and bad>=10):
                save(folder/'result.json',dict(state='complete',mode=mode,selection='development macro Top5',best=load(folder/'best_validation.json'),clinical_test_used=False,updated_at_utc=now()));decision[0]=1
            elif load(WORK/'fusion_resize.json').get('world_size',world)!=world:decision[0]=2
        dist.broadcast(decision,src=0);dist.barrier()
        if decision.item():break
    dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'fusion'/sys.argv[1]/('rank_'+os.environ.get('RANK','0')+'.failure.json'),dict(error_type=type(e).__name__,updated_at_utc=now()));raise
