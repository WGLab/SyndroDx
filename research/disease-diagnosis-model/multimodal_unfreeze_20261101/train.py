"""Matched patient-level visual/HPO continuation with epoch-boundary recovery."""
import fcntl
import gc
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from data import arrays,Patients,collate,macro_hits,retrieval,fewshot
from model import JointModel


def save_checkpoint(path,state):
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);tmp.replace(path)


def linked_copy(source,dest):
    dest.unlink(missing_ok=True);os.link(source,dest)


@torch.no_grad()
def evaluate(model,data,groups,protocol,controls=False):
    model.eval();train,val=data['train'],data['validation'];active=np.r_[train,val]
    z=np.zeros_like(data['features'])
    loader=DataLoader(Patients(active,groups),batch_size=16,num_workers=2,collate_fn=collate,pin_memory=True)
    # FP32 visual inference matches the original frozen feature pipeline.
    for ix,images,counts in loader:
        z[ix]=model.patient_features(images.cuda(non_blocking=True),counts,protocol['image_chunk']).cpu().numpy()
    def diagnosis(limit=None,shuffled=False):
        out=[]
        source=val.copy()
        if shuffled:
            rng=np.random.default_rng(20261101);order=rng.permutation(len(val));permutation=np.empty(len(val),int);permutation[order]=np.roll(order,1);source=val[permutation]
        for start in range(0,len(val),64):
            ix=val[start:start+64];ids=data['ids'][ix].copy();states=data['states'][ix]
            if limit is not None:
                for row,index in enumerate(ix):
                    positive=np.flatnonzero((ids[row]!=0)&(states[row]==0))
                    rng=np.random.default_rng(20261031+int(index))
                    keep=set(rng.choice(positive,min(limit,len(positive)),replace=False))
                    for j in positive:
                        if j not in keep:ids[row,j]=0
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_=model.fusion(torch.from_numpy(z[source[start:start+64]]).cuda(),torch.from_numpy(ids).cuda(),torch.from_numpy(states).cuda())
            out.append(logits.float().cpu().numpy())
        logits=np.concatenate(out);metrics,hits=macro_hits(logits,data['labels'][val]);return metrics,hits,logits
    diag,diag_hits,logits=diagnosis();ret,ret_hits=retrieval(z,train,val,data['labels'])
    metrics=dict(diagnosis=diag,native_visual_retrieval=ret,
        native_feature_mean_cosine_to_initial=float(np.sum(z[val]*data['features'][val],axis=1).mean()))
    if controls:
        metrics['controls']={f'HPO_{k}':diagnosis(k)[0] for k in [1,3,5]}
        metrics['controls']['shuffled_face']=diagnosis(shuffled=True)[0]
        metrics['supervised_disease_fewshot']=fewshot(z,train,val,data['labels'])
    model.train()
    return metrics,dict(diagnosis=diag_hits[1],retrieval=ret_hits[1]),logits,z


def supported_decline(current,initial,labels):
    # Paired disease-stratum bootstrap; no test set is involved.
    delta=current.astype(float)-initial.astype(float)
    per_disease=np.array([delta[labels==c].mean() for c in np.unique(labels)])
    rng=np.random.default_rng(20261101)
    samples=per_disease[rng.integers(0,len(per_disease),(2000,len(per_disease)))].mean(1)
    ci=np.quantile(samples,[.025,.975]).tolist()
    return ci[1]<-.02,dict(macro_delta=float(per_disease.mean()),paired_disease_bootstrap_ci95=ci)


def main(arm,seed):
    folder=WORK/'cases'/f'{arm}__seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(folder/'result.json').get('state')=='complete':return
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    for asset in load(WORK/'assets.json').values():assert digest(asset['path'])==asset['sha256']
    cache=load(WORK/'cache_receipt.json');assert cache['state']=='passed'
    assert digest(WORK/'private/images_rgb112.npy')==cache['images_sha256']
    assert digest(WORK/'private/patient_image_offsets.json')==cache['offsets_sha256']
    smoke=load(WORK/'smoke.json');assert smoke['state']=='passed';micro=smoke['micro_patients']
    p=load(WORK/'protocol.json');assert arm in p['arms'] and seed in p['seeds']
    torch.manual_seed(seed);np.random.seed(seed)
    data,groups=arrays();train,val=data['train'],data['validation'];labels=data['labels']
    r=load(WORK/'data_receipt.json');assert digest(r['profiles_path'])==r['profiles_sha256']
    with np.load(r['profiles_path']) as d:profiles={k:torch.from_numpy(np.array(d[k])).cuda() for k in d.files}
    model=JointModel(arm).cuda()
    opt_groups=[dict(params=[v for v in model.fusion.parameters() if v.requires_grad],lr=p['fusion_learning_rate'],base_lr=p['fusion_learning_rate'])]
    vision=[v for v in model.vision.parameters() if v.requires_grad]
    if vision:opt_groups.append(dict(params=vision,lr=p['visual_learning_rate'],base_lr=p['visual_learning_rate']))
    optimizer=torch.optim.AdamW(opt_groups,weight_decay=p['weight_decay'])
    weights=torch.bincount(torch.from_numpy(labels[train]).cuda(),minlength=r['classes']).float().clamp_min(1).rsqrt();weights/=weights.mean()
    epoch=step=bad=declines=0;best=(-1.,-1.);best_epoch=0;initial_hits=None
    if (folder/'last.pt').exists():
        ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==digest(WORK/'protocol.json') and ck['data_sha256']==digest(WORK/'data_receipt.json')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        epoch,step,bad,declines,best,best_epoch=[ck[k] for k in ['epoch','step','bad','declines','best','best_epoch']]
        initial_hits=ck['initial_hits'];del ck
    def snapshot(epoch,metrics):
        save_checkpoint(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,step=step,
            bad=bad,declines=declines,best=best,best_epoch=best_epoch,initial_hits=initial_hits,
            arm=arm,seed=seed,validation=metrics,protocol_sha256=digest(WORK/'protocol.json'),data_sha256=digest(WORK/'data_receipt.json')))
    if initial_hits is None:
        save(folder/'status.json',dict(state='epoch0_validation',epoch=0,updated_at_utc=now()))
        baseline,initial_hits,_,_=evaluate(model,data,groups,p,controls=True)
        best=(baseline['diagnosis']['top1'],baseline['diagnosis']['top5'])
        save(folder/'epoch0.json',baseline);snapshot(0,baseline);linked_copy(folder/'last.pt',folder/'best.pt')
    stop='maximum_epochs'
    for epoch in range(epoch+1,p['max_epochs']+1):
        began=time.monotonic();model.train();losses=[]
        order=np.random.default_rng(seed+epoch*100003).permutation(train)
        for offset in range(0,len(order),p['global_patient_batch']):
            batch=order[offset:offset+p['global_patient_batch']]
            torch.manual_seed(seed*1000003+epoch*1009+offset)
            fraction=(epoch-1+offset/len(order))/p['max_epochs']
            warm=min(1.,(epoch-1+(offset+len(batch))/len(order))/p['warmup_epochs'])
            scale=warm*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*scale
            denominator=weights[torch.from_numpy(labels[batch]).cuda()].sum()
            optimizer.zero_grad(set_to_none=True);total=0.
            loader=DataLoader(Patients(batch,groups),batch_size=micro,num_workers=0,collate_fn=collate)
            for ix,images,counts in loader:
                ids=torch.from_numpy(data['ids'][ix]).cuda();states=torch.from_numpy(data['states'][ix]).cuda();y=torch.from_numpy(labels[ix]).cuda()
                dropped=ids.clone();dropped[torch.rand(dropped.shape,device='cuda')<p['HPO_dropout']]=0
                drop=torch.rand(len(ix),device='cuda')
                dropped[(drop>=p['face_dropout'])&(drop<p['face_dropout']+p['HPO_modality_dropout'])]=0
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    features=model.patient_features(images.cuda(non_blocking=True),counts,p['image_chunk'])
                    features=features*(drop>=p['face_dropout'])[:,None]
                    logits,representation=model.fusion(features,dropped,states)
                    primary=F.cross_entropy(logits.float(),y,weight=weights,reduction='sum')/denominator
                    h,_,_=model.fusion.phenotype(ids,states)
                    auxiliary=F.cross_entropy(model.fusion.head(h).float(),y,weight=weights,reduction='sum')/denominator
                    knowledge=model.profile_loss(model.fusion,representation,y,profiles)
                    loss=primary+.5*auxiliary+.1*knowledge*(len(ix)/len(batch))
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite_training_loss')
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();step+=1;losses.append(total)
            save(folder/'status.json',dict(state='training',epoch=epoch,step=step,patients_this_epoch=min(offset+len(batch),len(order)),train_patients=len(order),loss=total,
                micro_patients=micro,peak_memory_GiB=torch.cuda.max_memory_allocated()/1024**3,updated_at_utc=now()))
        metrics,hits,_,_=evaluate(model,data,groups,p)
        score=(metrics['diagnosis']['top1'],metrics['diagnosis']['top5'])
        improved=score>tuple(best)
        if improved:best,best_epoch,bad=score,epoch,0
        else:bad+=1
        guards={};adverse=[]
        for name in ['diagnosis','retrieval']:
            drop,guards[name]=supported_decline(hits[name],initial_hits[name],labels[val]);adverse.append(drop)
        declines=declines+1 if any(adverse) else 0
        snapshot(epoch,metrics)
        if improved:linked_copy(folder/'last.pt',folder/'best.pt')
        report=dict(state='epoch_complete',epoch=epoch,step=step,loss=float(np.mean(losses)),validation=metrics,
            guard=guards,best_epoch=best_epoch,epoch_seconds=time.monotonic()-began,updated_at_utc=now())
        save(folder/'history'/f'epoch_{epoch:03d}.json',report);save(folder/'status.json',report)
        if declines>=2:stop='two_supported_development_declines';break
        if epoch>=p['min_epochs'] and bad>=p['patience']:stop='development_plateau';break
    # Release optimizer states before loading the selected model checkpoint.
    del optimizer;gc.collect();torch.cuda.empty_cache()
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model']);selected_epoch=selected['epoch'];del selected
    final,hits,logits,features=evaluate(model,data,groups,p,controls=True)
    np.savez(folder/'outputs.private.npz',logits=logits,features=features,validation_labels=labels[val],**hits)
    result=dict(state='complete',arm=arm,seed=seed,selected_epoch=selected_epoch,completed_epoch=epoch,stop_reason=stop,
        metrics=final,epoch0=load(folder/'epoch0.json'),checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=digest(folder/'best.pt'),
        parameter_counts=model.parameter_counts() if hasattr(model,'parameter_counts') else dict(visual_trainable=sum(q.numel() for q in model.vision.parameters() if q.requires_grad)),
        test_or_expert_used=False,release_modified=False,updated_at_utc=now())
    save(folder/'result.json',result);save(folder/'status.json',dict(state='complete',epoch=epoch,selected_epoch=selected_epoch,updated_at_utc=now()))


if __name__=='__main__':
    arm,seed=sys.argv[1],int(sys.argv[2]);guarded(lambda:main(arm,seed),WORK/'cases'/f'{arm}__seed{seed}'/'failure.json')
