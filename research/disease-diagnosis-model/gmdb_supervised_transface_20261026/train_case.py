"""One independent, resumable, disease-supervised comparison case per GPU."""
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import WORK,load,save,digest,now,verify_code
from data import Images,verify_data,validation_metrics
from disease_model import DiseaseModel

def random_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
def save_torch(path,state):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    return digest(path)

@torch.no_grad()
def validate(model,loader,records):
    model.eval();predictions=[];offsets=[];loss=0.;count=0
    for images,labels,indices in loader:
        images=images.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(images)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite_development_logits')
        loss+=float(F.cross_entropy(logits.float(),labels,reduction='sum'));count+=len(labels)
        predictions.append(logits.float().cpu().numpy());offsets.extend(indices.tolist())
    metrics=validation_metrics(np.concatenate(predictions),offsets,records);metrics['cross_entropy']=loss/count
    return metrics

def main(case_name):
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    verify_code();receipt=verify_data();p=load(WORK/'protocol.json')['stage2']
    cases=load(WORK/'cases_manifest.json')['cases']
    matches=[c for c in cases if c['name']==case_name]
    if len(matches)!=1:raise RuntimeError('unknown_training_case')
    case=matches[0];folder=WORK/'cases'/case_name;folder.mkdir(exist_ok=True)
    if load(folder/'result.json').get('state')=='complete':return
    seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train_indices=load(WORK/'private'/f"budget_{case['budget']}.private.json")['indices']
    records=load(WORK/'private/records.private.json')['rows']
    val_indices=[r['array_index'] for r in records if r['split']=='validation']
    counts=np.bincount([records[i]['class_index'] for i in train_indices],minlength=receipt['diseases'])
    if np.any(counts==0):raise RuntimeError('training_case_missing_disease')
    weights=torch.tensor(.5+.5*counts.min()/counts,dtype=torch.float32,device='cuda')
    selection=load(WORK/'selected/identity_checkpoint.json')
    model=DiseaseModel(receipt['diseases'],case['arm'],selection).cuda()
    optimizer=torch.optim.Adam([
        dict(params=model.face.feature.parameters(),lr=case['lr'],weight_decay=p['feature_weight_decay']),
        dict(params=model.classifier.parameters(),lr=case['lr'],weight_decay=p['classifier_weight_decay'])])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5,min_lr=1e-5,threshold=5e-4)
    signature=dict(case=case,protocol_sha256=digest(WORK/'protocol.json'),data_files=receipt['files'],
        identity_checkpoint_sha256=selection.get('sha256') if case['arm']=='identity_cpt' else None)
    start=0;best=None;steps=0
    resume=load(folder/'latest.json')
    if resume:
        path=folder/'latest.pt'
        if digest(path)!=resume['sha256']:raise RuntimeError('case_checkpoint_checksum_mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise RuntimeError('case_checkpoint_configuration_mismatch')
        model.load_adaptation(state['adaptation']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];best=state['best'];steps=state['steps'];restore_rng(state['rng']);del state
    recorded_best=load(folder/'best.json')
    if recorded_best and (best is None or tuple(recorded_best['score'])>tuple(best['score'])):
        if digest(folder/'best.pt')!=recorded_best['sha256']:raise RuntimeError('best_checkpoint_checksum_mismatch')
        best_state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False,mmap=True)
        if best_state['signature']!=signature:raise RuntimeError('best_checkpoint_configuration_mismatch')
        best=recorded_best;del best_state
    batch=load(folder/'batch_override.json').get('batch_size',p['batch_size'])
    train_dataset=Images(train_indices,train=True);val_dataset=Images(val_indices)
    train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=2,pin_memory=True,
        persistent_workers=True,drop_last=True,prefetch_factor=2)
    val_loader=DataLoader(val_dataset,batch_size=256,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    for epoch in range(start,p['epochs']):
        model.train();total=0.;count=0;begun=time.monotonic()
        for images,labels,_ in train_loader:
            images=images.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(images)
            loss=F.cross_entropy(logits.float(),labels,weight=weights)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_disease_training_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],3.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(labels);count+=len(labels);steps+=1
            if steps%20==0:save(folder/'status.json',dict(state='training',case=case_name,epoch=epoch+1,steps=steps,
                loss=float(loss.detach()),batch_size=batch,updated_at_utc=now()))
        save(folder/'status.json',dict(state='development_validation',epoch=epoch+1,steps=steps,updated_at_utc=now()))
        metrics=validate(model,val_loader,records);score=(metrics['patient_macro']['top5'],metrics['patient_macro']['top1'])
        scheduler.step(score[0])
        row=dict(epoch=epoch+1,steps=steps,training_loss=total/count,validation=metrics,
            learning_rates=[g['lr'] for g in optimizer.param_groups],epoch_seconds=time.monotonic()-begun,updated_at_utc=now())
        with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if best is None or score>tuple(best['score']):
            sha=save_torch(folder/'best.pt',dict(adaptation=model.adaptation_state(),signature=signature,epoch=epoch+1,validation=metrics))
            best=dict(epoch=epoch+1,score=list(score),sha256=sha,path=str(folder/'best.pt'),validation=metrics)
            save(folder/'best.json',best)
        if (epoch+1)%5==0 or epoch+1==p['epochs']:
            sha=save_torch(folder/'latest.pt',dict(adaptation=model.adaptation_state(),signature=signature,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,steps=steps,best=best,rng=random_state()))
            save(folder/'latest.json',dict(epoch=epoch+1,sha256=sha,updated_at_utc=now()))
    save(folder/'result.json',dict(state='complete',case=case,epochs=p['epochs'],best=load(folder/'best.json'),
        training_images=len(train_indices),validation_images=len(val_indices),test_used=False,updated_at_utc=now()))
    save(folder/'status.json',dict(state='complete',epoch=p['epochs'],steps=steps,updated_at_utc=now()))

if __name__=='__main__':
    name=sys.argv[1]
    try:main(name)
    except Exception as error:
        save(WORK/'cases'/name/'failure.json',dict(error_type=type(error).__name__,oom=isinstance(error,torch.cuda.OutOfMemoryError),updated_at_utc=now()))
        raise
