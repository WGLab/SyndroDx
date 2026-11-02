"""Full-backbone disease ArcFace adaptation; selection uses development only."""
import os,sys,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from data import Images,verify_data,macro_hits
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from disease_model import Model
from model import fixed_batchnorm

@torch.no_grad()
def validation(model,rows):
    model.eval();parts={}
    for split in ['train','validation']:
        ids=[i for i,r in enumerate(rows) if r['split']==split and r['class_index']>=0];features=[]
        for x,_,_ in DataLoader(Images(ids),batch_size=256,num_workers=2,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):f=model.embed(x.cuda(non_blocking=True))
            features.append(f.cpu().numpy())
        grouped={}
        for i,f in zip(ids,np.concatenate(features)):grouped.setdefault(rows[i]['subject'],[]).append((rows[i]['class_index'],f))
        x=[];y=[]
        for entries in grouped.values():
            assert len({v[0] for v in entries})==1
            v=np.mean([v[1] for v in entries],axis=0);x.append(v/max(np.linalg.norm(v),1e-12));y.append(entries[0][0])
        parts[split]=(np.array(x),np.array(y))
    x,y=parts['train'];q,qy=parts['validation'];proto=np.stack([x[y==i].mean(0) for i in range(int(y.max())+1)])
    proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
    return dict(macro_hits(q@proto.T,qy),patients=len(q),metric='patient-balanced disease prototype cosine retrieval')

def main():
    os.umask(0o077);arm=sys.argv[1];lr=float(sys.argv[2]);seed=int(sys.argv[3]);name=sys.argv[4];folder=WORK/name;folder.mkdir(exist_ok=True)
    p=load(WORK/'protocol.json');receipt=verify_data();rows=load(WORK/'private/records.private.json')['rows']
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation' and r['class_index']>=0]
    assert not {rows[i]['subject'] for i in train}&{rows[i]['subject'] for i in val}
    assert len({rows[i]['class_index'] for i in train})==receipt['diseases']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);torch.cuda.set_device(0)
    model=Model('arcface_identity',dict(num_identity_classes=receipt['diseases'],arcface_scale=64.,arcface_margin=.5)).cuda()
    selection=load(WORK/'initialization.json')[arm]
    if arm=='chop_cpt':
        assert digest(selection['path'])==selection['sha256']
        model.backbone.load_state_dict(torch.load(selection['path'],map_location='cpu',weights_only=False)['backbone'],strict=True)
    model.configure_feature_only()
    opt=torch.optim.Adam([dict(params=model.backbone.encoder.feature.parameters(),lr=lr,weight_decay=0.),dict(params=model.head.parameters(),lr=lr,weight_decay=.0005)])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-5)
    counts=np.bincount([rows[i]['class_index'] for i in train],minlength=receipt['diseases'])
    class_weights=torch.tensor(.5+.5*counts.min()/counts,dtype=torch.float32,device='cuda')
    begin=0;best=(-1.,-1.)
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);begin=state['epoch'];best=tuple(state['best']);scheduler.load_state_dict(state['scheduler']) if 'scheduler' in state else None
    loader=DataLoader(Images(train,train=True),batch_size=p['batch_size'],shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
    for epoch in range(begin,p['epochs']):
        model.train();losses=[]
        for x,y,_ in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda(non_blocking=True),y.cuda(non_blocking=True))
            loss=F.cross_entropy(logits.float(),y.cuda(non_blocking=True),weight=class_weights)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_disease_loss')
            loss.backward()
            if epoch<p['head_warmup_epochs']:
                for v in model.backbone.parameters():v.grad=None
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        save(folder/'status.json',dict(state='training',epoch=epoch+1,loss=float(np.mean(losses)),arm=arm,seed=seed,backbone_lr=lr,updated_at_utc=now()))
        if (epoch+1)%5==0:
            metrics=validation(model,rows);score=(metrics['top5'],metrics['top1']);scheduler.step(score[0])
            if score>best:
                best=score;temp=folder/'best.tmp';torch.save(dict(backbone=model.backbone.state_dict(),head=model.head.state_dict(),epoch=epoch+1),temp);temp.replace(folder/'best.pt')
                save(folder/'best.json',dict(path=str(folder/'best.pt'),sha256=digest(folder/'best.pt'),epoch=epoch+1,metrics=metrics,seed=seed,arm=arm,lr=lr))
            temp=folder/'last.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,best=best),temp);temp.replace(folder/'last.pt')
            save(folder/'latest_validation.json',dict(epoch=epoch+1,metrics=metrics,updated_at_utc=now()))
    save(folder/'result.json',dict(state='complete',best=load(folder/'best.json'),test_used=False,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/sys.argv[4]/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
