"""Offline BLIP-2 Stage-1 pilot with epoch checkpoints and paired controls."""
import fcntl
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from model import Stage1

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def group_mean(values,labels):return float(np.mean([values[labels==c].mean() for c in np.unique(labels)]))
def ranking(sim,truth,gallery_truth):
    order=np.argsort(-sim,axis=1,kind='stable')
    hits={str(k):(gallery_truth[order[:,:k]]==truth[:,None]).any(1) for k in [1,5,10]}
    return {'top'+k:group_mean(h,truth) for k,h in hits.items()},hits

@torch.no_grad()
def encode(model,data,indices):
    model.eval();vi=[];tx=[]
    for start in range(0,len(indices),32):
        ix=indices[start:start+32]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,_,image=model.visual(data['tokens'][ix],data['mask'][ix])
            text=model.text(data['input_ids'][ix],data['attention_mask'][ix])
        vi.append(image.float().cpu().numpy());tx.append(text.float().cpu().numpy())
    return np.concatenate(vi),np.concatenate(tx)

def alignment_metrics(image,text,groups):
    unique,indices=np.unique(groups,return_index=True)
    # Each unique caption occurs once in the candidate text gallery.
    sim=np.einsum('bqd,nd->bnq',image,text[indices]).max(-1)
    a,ah=ranking(sim,groups,unique)
    b,bh=ranking(sim.T,unique,groups)
    return dict(image_to_text=a,text_to_image=b,mean_R1=(a['top1']+b['top1'])/2,
        mean_R5=(a['top5']+b['top5'])/2,patients=len(groups),caption_groups=len(unique)),dict(i2t=ah,t2i=bh)

def save_checkpoint(path,state):
    tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);tmp.replace(path)

def main(condition,seed):
    os.umask(0o077);torch.set_num_threads(4)
    folder=WORK/'cases'/f'{condition}__seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(folder/'result.json').get('state')=='complete':return
    p=load(WORK/'protocol.json');r=load(WORK/'data_receipt.json')
    assert condition in p['conditions'] and seed in p['seeds']
    assert digest(WORK/'private/paired.npz')==r['paired_sha256']
    assert digest(r['features']['tokens_path'])==r['features']['tokens_sha256']
    assert digest(r['mask_path'])==r['features']['mask_sha256']
    assert load(WORK/'tests.json')['state']=='passed'
    with np.load(WORK/'private/paired.npz') as f:arr={k:f[k].copy() for k in f.files}
    train,val=arr['train'],arr['validation'];members=arr['members']
    data={k:torch.from_numpy(arr[k]).cuda() for k in ['input_ids','attention_mask','groups','present']}
    raw=np.load(r['features']['tokens_path'],mmap_mode='r')
    data['tokens']=torch.from_numpy(np.array(raw[members]).reshape(len(members),-1,768)).cuda();del raw
    rawmask=np.load(r['mask_path']);data['mask']=torch.from_numpy(rawmask[members].reshape(len(members),-1)).cuda()
    torch.manual_seed(seed);np.random.seed(seed)
    model=Stage1().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],betas=(.9,.98))
    save(folder/'initialization.json',dict(**model.initialization_report,parameters=sum(v.numel() for v in model.parameters()),updated_at_utc=now()))
    # No disease labels enter the model or losses. They are used only in the final retrieval evaluation.
    face_map=np.arange(len(members))
    if condition=='shuffled':
        rng=np.random.default_rng(20261101+seed)
        unique,count=np.unique(arr['groups'][train],return_counts=True)
        assert 2*count.max()<=len(train)
        order=np.concatenate([rng.permutation(train[arr['groups'][train]==g]) for g in rng.permutation(unique)])
        candidate=np.roll(order,int(count.max()))
        assert np.all(arr['groups'][order]!=arr['groups'][candidate])
        face_map[order]=candidate
        assert np.all(face_map[train]!=train) and np.array_equal(face_map[val],val)
    epoch,step,best,bad=0,0,(-1.,-1.),0
    checkpoint_path=folder/'last.pt'
    if checkpoint_path.exists():
        s=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert s['protocol_sha256']==digest(WORK/'protocol.json') and s['data_sha256']==digest(WORK/'data_receipt.json')
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer'])
        epoch,step,best,bad=s['epoch'],s['step'],tuple(s['best']),s['bad']
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng']);del s
    def snapshot(e):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=e,step=step,best=best,bad=bad,
        condition=condition,seed=seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        protocol_sha256=digest(WORK/'protocol.json'),data_sha256=digest(WORK/'data_receipt.json'))
    if not checkpoint_path.exists():
        image,text=encode(model,data,val);metrics,_=alignment_metrics(image,text,arr['groups'][val])
        best=(metrics['mean_R1'],metrics['mean_R5'])
        save_checkpoint(checkpoint_path,snapshot(0));os.link(checkpoint_path,folder/'best.pt')
        save(folder/'initial_validation.json',dict(metrics=metrics,updated_at_utc=now()))
    total_steps=math.ceil(len(train)/p['batch_size'])*p['max_epochs'];started_all=time.monotonic()
    for epoch in range(epoch+1,p['max_epochs']+1):
        model.train();start_time=time.monotonic();losses=[];nneg=0
        order=np.random.default_rng(seed+1009*epoch).permutation(train)
        batches=list(np.array_split(order,math.ceil(len(order)/p['batch_size'])))
        for ix in batches:
            fx=face_map[ix]
            progress=max(0.,(step-p['warmup_steps'])/max(1,total_steps-p['warmup_steps']))
            lr=p['min_lr']+(p['learning_rate']-p['min_lr'])*.5*(1+math.cos(math.pi*min(1,progress)))
            lr*=min(1.,(step+1)/p['warmup_steps'])
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,parts=model(data['tokens'][fx],data['mask'][fx],data['input_ids'][ix],data['attention_mask'][ix],data['groups'][ix],data['present'][ix])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();step+=1;nneg+=parts['negative_rows']
            losses.append([float(loss.detach())]+[float(parts[k]) for k in ['itc','itm','itg']])
            if step%10==0:
                save(folder/'status.json',dict(state='training',epoch=epoch,step=step,
                    loss=losses[-1][0],ITC=losses[-1][1],ITM=losses[-1][2],ITG=losses[-1][3],lr=lr,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,updated_at_utc=now()))
        image,text=encode(model,data,val);metrics,_=alignment_metrics(image,text,arr['groups'][val])
        score=(metrics['mean_R1'],metrics['mean_R5']);improved=score>best
        if improved:best,bad=score,0
        else:bad+=1
        save_checkpoint(checkpoint_path,snapshot(epoch))
        if improved:
            (folder/'best.pt').unlink(missing_ok=True);os.link(checkpoint_path,folder/'best.pt')
        average=np.mean(losses,axis=0)
        report=dict(state='training',epoch=epoch,step=step,loss=float(average[0]),ITC=float(average[1]),ITM=float(average[2]),ITG=float(average[3]),
            validation=metrics,best_mean_R1=best[0],itm_negative_eligible_fraction=nneg/len(train),
            epoch_seconds=time.monotonic()-start_time,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,updated_at_utc=now())
        save(folder/'status.json',report)
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        if epoch>=p['min_epochs'] and bad>=p['patience']:break
    s=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);selected=s['epoch'];del s
    image,text=encode(model,data,np.arange(len(members)))
    metrics,align_hits=alignment_metrics(image[val],text[val],arr['groups'][val])
    pooled=norm(image.mean(1));native=norm(arr['native_features'])
    disease,dh=ranking(pooled[val]@pooled[train].T,arr['labels'][val],arr['labels'][train])
    base,bh=ranking(native[val]@native[train].T,arr['labels'][val],arr['labels'][train])
    # Within the same validation cohort, random image-caption pairing must not improve meaningful alignment.
    order=np.random.default_rng(seed+41).permutation(len(val));perm=np.empty(len(val),int);perm[order]=np.roll(order,1)
    shuffled,_=alignment_metrics(image[val][perm],text[val],arr['groups'][val])
    storage={'labels':arr['labels'][val],'caption_groups':arr['groups'][val],
        'query_embeddings':pooled[val],'text_embeddings':text[val]}
    for tag,h in [('query',dh),('native',bh)]:
        for k,v in h.items():storage[f'{tag}_disease_top{k}']=v
    for direction,hs in align_hits.items():
        for k,v in hs.items():storage[f'{direction}_top{k}']=v
    np.savez(folder/'evaluation.private.npz',**storage)
    assert digest(r['visual']['path'])==r['visual']['sha256']
    result=dict(state='complete',condition=condition,seed=seed,selected_epoch=selected,completed_epoch=epoch,
        alignment=metrics,shuffled_validation_alignment=shuffled,image_only_disease_retrieval=disease,
        native_visual_disease_retrieval=base,checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=digest(folder/'best.pt'),
        native_encoder_unchanged=True,clinical_test_used=False,supervised_disease_loss_used=False,
        runtime_seconds=time.monotonic()-started_all,updated_at_utc=now())
    save(folder/'result.json',result);save(folder/'status.json',dict(state='complete',epoch=epoch,updated_at_utc=now()))

if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except Exception as e:
        save(WORK/'cases'/f'{sys.argv[1]}__seed{sys.argv[2]}'/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
