import os,sys,random
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from data import verify_data,Images,macro_hits,validation_metrics
from network import R50
from lib.utils_functions import seed_worker

@torch.no_grad()
def validate(model,ids,rows,weights):
    model.eval();scores=[];indices=[];total=0.
    for x,y,ix in DataLoader(Images(ids),batch_size=128,num_workers=2,pin_memory=True):
        logits,_=model(x.cuda(non_blocking=True));total+=float(F.cross_entropy(logits,y.cuda(),weight=weights,reduction='sum'));scores.append(logits.cpu().numpy());indices.extend(ix.tolist())
    r=validation_metrics(np.concatenate(scores),indices,rows);r['loss']=total/len(indices);model.train();return r

def main():
    os.umask(0o077);arm=sys.argv[1];seed=int(sys.argv[2]);folder=WORK/(arm+'_'+str(seed));folder.mkdir(exist_ok=True)
    if load(folder/'result.json').get('state')=='complete':return
    p=load(WORK/'protocol.json');receipt=verify_data();rows=load(WORK/'private/records.private.json')['rows'];tr=[i for i,r in enumerate(rows) if r['split']=='train'];va=[i for i,r in enumerate(rows) if r['split']=='validation']
    torch.set_num_threads(4);torch.cuda.set_device(0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=R50(receipt['diseases'],arm).cuda();opt=torch.optim.Adam([dict(params=m.features.parameters(),lr=p['feature_lr'],weight_decay=p['feature_weight_decay']),dict(params=m.classifier.parameters(),lr=p['classifier_lr'],weight_decay=p['classifier_weight_decay'])])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,threshold=5e-4,min_lr=1e-5)
    count=np.bincount([rows[i]['class_index'] for i in tr],minlength=receipt['diseases']);weights=torch.tensor(.5+.5*count.min()/count,dtype=torch.float32,device='cuda')
    start=0;best=-1.
    if (folder/'last.pt').exists():
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert s['protocol_sha256']==digest(WORK/'protocol.json');m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sched.load_state_dict(s['scheduler']);start=s['epoch'];best=s['best']
        if 'rng' in s:
            torch.set_rng_state(s['rng']['torch']);torch.cuda.set_rng_state(s['rng']['cuda']);np.random.set_state(s['rng']['numpy']);random.setstate(s['rng']['python'])
    loader=DataLoader(Images(tr,train=True),batch_size=p['batch_size'],shuffle=True,num_workers=2,pin_memory=True,drop_last=True,worker_init_fn=seed_worker)
    if start>=p['epochs']:
        report=validate(m,va,rows,weights)
    for epoch in range(start,p['epochs']):
        m.train();losses=[]
        for x,y,_ in loader:
            opt.zero_grad(set_to_none=True);logits,_=m(x.cuda(non_blocking=True));loss=F.cross_entropy(logits,y.cuda(non_blocking=True),weight=weights)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        report=validate(m,va,rows,weights);score=report['image_macro']['top5'];sched.step(score)
        improved=score>best;best=max(best,score)
        state=dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch+1,best=best,rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),protocol_sha256=digest(WORK/'protocol.json'))
        temp=folder/'last.tmp';torch.save(state,temp);temp.replace(folder/'last.pt')
        if improved:
            target=folder/'best.pt';target.unlink(missing_ok=True);os.link(folder/'last.pt',target);save(folder/'best.json',dict(epoch=epoch+1,validation=report,path=str(target),sha256=digest(target)))
        save(folder/'status.json',dict(state='training',epoch=epoch+1,loss=float(np.mean(losses)),validation=report,updated_at_utc=now()))
    save(folder/'result.json',dict(state='complete',arm=arm,seed=seed,primary_checkpoint=str(folder/'last.pt'),primary_validation=report,secondary_best=load(folder/'best.json'),test_used_for_training=False,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/(sys.argv[1]+'_'+sys.argv[2])/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
