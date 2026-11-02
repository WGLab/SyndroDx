"""Matched adapter-only experiments; patient predictions remain private."""
import fcntl
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from hpo_model import PhenotypeModel
from query_model import QueryFusion

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def hits(scores,truth,gallery=None):
    order=np.argsort(-scores,axis=1,kind='stable')
    return {str(k):(order[:,:k]==truth[:,None]).any(1) if gallery is None else (gallery[order[:,:k]]==truth[:,None]).any(1) for k in [1,5,10]}
def metrics(hit,truth):return {'top'+k:float(np.mean([v[truth==c].mean() for c in np.unique(truth)])) for k,v in hit.items()}

@torch.no_grad()
def predict(model,data,indices,face_indices=None,term_limit=None):
    model.eval();ss=[];zz=[]
    fx=indices if face_indices is None else face_indices
    for start in range(0,len(indices),64):
        ix=indices[start:start+64];jx=fx[start:start+64]
        ids=data['ids'][ix].clone();states=data['states'][ix]
        if term_limit is not None:
            for j,index in enumerate(ix):
                positive=torch.where(ids[j].ne(0)&states[j].eq(0))[0]
                choice=np.random.default_rng(20261101+int(index)).permutation(len(positive))[term_limit:]
                ids[j,positive[choice]]=0
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s,z=model(data['features'][jx],data['tokens'][jx],data['mask'][jx],ids,states)
        ss.append(s.float().cpu().numpy());zz.append(z.float().cpu().numpy())
    return np.concatenate(ss),normalize(np.concatenate(zz))

def atomic_checkpoint(path,state):
    temp=path.with_suffix('.tmp.pt');torch.save(state,temp);temp.replace(path)

def main(mode,seed):
    os.umask(0o077);torch.set_num_threads(4)
    folder=WORK/'cases'/f'{mode}__seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(folder/'result.json').get('state')=='complete':return
    p=load(WORK/'protocol.json');r=load(WORK/'data_receipt.json');fr=load(WORK/'features/receipt.json')
    assert mode in p['modes'] and seed in p['seeds'] and fr['state']=='complete'
    assert digest(r['gmdb_path'])==r['gmdb_sha256']
    assert digest(r['term_embeddings_path'])==r['term_embeddings_sha256']
    assert digest(fr['tokens_path'])==fr['tokens_sha256']
    assert digest(WORK/'private/local_mask.npy')==fr['mask_sha256']
    pin=r['hpo'][str(seed)];assert digest(pin['path'])==pin['sha256']
    torch.manual_seed(seed);np.random.seed(seed)
    with np.load(r['gmdb_path']) as f:arr={k:f[k].copy() for k in f.files}
    train,val=arr['train'],arr['validation'];truth=arr['labels'][val]
    data={k:torch.from_numpy(arr[k]).cuda() for k in ['features','ids','states','labels']}
    cache=np.load(fr['tokens_path'],mmap_mode='r');n=len(cache)
    data['tokens']=torch.from_numpy(np.array(cache).reshape(n,-1,768)).cuda();del cache
    data['mask']=torch.from_numpy(np.load(WORK/'private/local_mask.npy').reshape(n,-1)).cuda()
    c=r['counts'];hpo=PhenotypeModel(torch.from_numpy(np.load(r['term_embeddings_path'])),c['GMDB_classes'],c['public_training_diseases'],'hpo')
    state=torch.load(pin['path'],map_location='cpu',weights_only=False);hpo.load_state_dict(state['model'],strict=True);del state
    model=QueryFusion(hpo,mode).cuda()
    frozen={k:v.detach().cpu().clone() for k,v in model.hpo.state_dict().items()}
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['learning_rate'],weight_decay=p['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4,min_lr=1e-6)
    weights=torch.bincount(data['labels'][train],minlength=c['GMDB_classes']).float().clamp_min(1).rsqrt();weights/=weights.mean()
    epoch,step,best,bad=0,0,(-1.,-1.),0
    if (folder/'last.pt').exists():
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert s['protocol_sha256']==digest(WORK/'protocol.json') and s['data_sha256']==digest(WORK/'data_receipt.json')
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        epoch,step,best,bad=s['epoch'],s['step'],tuple(s['best']),s['bad']
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng']);del s
    elif not (folder/'initial_validation.json').exists():
        scores,_=predict(model,data,val)
        save(folder/'initial_validation.json',dict(validation=metrics(hits(scores,truth),truth),initial_residual_zero=True,updated_at_utc=now()))
    started_all=time.monotonic()
    for epoch in range(epoch+1,p['max_epochs']+1):
        started=time.monotonic();model.train();losses=[]
        order=np.random.default_rng(seed+1009*epoch).permutation(train)
        for start in range(0,len(order),p['batch_size']):
            ix=order[start:start+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_=model(data['features'][ix],data['tokens'][ix],data['mask'][ix],data['ids'][ix],data['states'][ix])
                loss=F.cross_entropy(logits.float(),data['labels'][ix],weight=weights)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_training_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
            assert all(v.grad is None for v in model.hpo.parameters())
            optimizer.step();step+=1;losses.append(float(loss.detach()))
        scores,_=predict(model,data,val);m=metrics(hits(scores,truth),truth);score=(m['top1'],m['top5'])
        scheduler.step(score[0]);improved=score>best
        if improved:best,bad=score,0
        else:bad+=1
        ckpt=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,step=step,best=best,bad=bad,mode=mode,seed=seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            protocol_sha256=digest(WORK/'protocol.json'),data_sha256=digest(WORK/'data_receipt.json'))
        atomic_checkpoint(folder/'last.pt',ckpt)
        if improved:
            (folder/'best.pt').unlink(missing_ok=True);os.link(folder/'last.pt',folder/'best.pt')
        report=dict(state='training',epoch=epoch,step=step,loss=float(np.mean(losses)),validation=m,best_validation_top1=best[0],
            epoch_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),updated_at_utc=now())
        save(folder/'status.json',report)
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        if epoch>=p['min_epochs'] and bad>=p['patience']:break
    s=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);selected=s['epoch'];del s
    allix=np.arange(n);scores,z=predict(model,data,allix)
    outcome={'diagnosis':hits(scores[val],truth),'retrieval':hits(z[val]@z[train].T,truth,arr['labels'][train])}
    # Frozen HPO and native face controls on exactly the same subjects/candidates.
    hs=[];hz=[]
    with torch.inference_mode():
        for start in range(0,n,64):
            ix=allix[start:start+64]
            with torch.autocast('cuda',dtype=torch.bfloat16):a,b=model.hpo(data['features'][ix],data['ids'][ix],data['states'][ix])
            hs.append(a.float().cpu());hz.append(b.float().cpu().numpy())
    hp=torch.cat(hs).softmax(-1).numpy();hv=normalize(np.concatenate(hz));visual=normalize(arr['features'])
    vpath=p['visual']['path'];assert digest(vpath)==p['visual']['sha256']
    vs=torch.load(vpath,map_location='cpu',weights_only=False)['model']
    classfile=PROJECT/'protected_visual_fusion_20261101/private/class_indices.json'
    assert digest(classfile)==load(PROJECT/'protected_visual_fusion_20261101/input_receipt.json')['class_indices_sha256']
    w=vs['head.weight'][load(classfile)['indices']].float();del vs
    vp=(64*torch.from_numpy(visual)@F.normalize(w,dim=1).T/4).softmax(-1).numpy()
    controls={};private={'labels':truth}
    for task,h in outcome.items():
        for k,v in h.items():private[f'query_{task}_{k}']=v
    for tag,prob,sim in [('hpo',hp[val],hv[val]@hv[train].T),('visual',vp[val],visual[val]@visual[train].T),
        ('late_fusion',.5*hp[val]+.5*vp[val],.5*(hv[val]@hv[train].T)+.5*(visual[val]@visual[train].T))]:
        outcomes={'diagnosis':hits(prob,truth),'retrieval':hits(sim,truth,arr['labels'][train])}
        controls[tag]={t:metrics(h,truth) for t,h in outcomes.items()}
        for task,h in outcomes.items():
            for k,v in h.items():private[f'{tag}_{task}_{k}']=v
    rng=np.random.default_rng(seed+20261101);order=rng.permutation(len(val));perm=np.empty(len(val),int);perm[order]=np.roll(order,1)
    assert np.all(perm!=np.arange(len(val)))
    sc,zz=predict(model,data,val,face_indices=val[perm]);controls['shuffled_face']={'diagnosis':metrics(hits(sc,truth),truth),'retrieval':metrics(hits(zz@z[train].T,truth,arr['labels'][train]),truth)}
    for limit in [1,3,5]:
        sc,_=predict(model,data,val,term_limit=limit);controls[f'HPO_{limit}']={'diagnosis':metrics(hits(sc,truth),truth)}
    assert all(torch.equal(v,model.hpo.state_dict()[k].detach().cpu()) for k,v in frozen.items())
    assert digest(pin['path'])==pin['sha256'] and digest(vpath)==p['visual']['sha256']
    np.savez(folder/'validation_hits.private.npz',**private)
    np.savez(folder/'validation_predictions.private.npz',logits=scores[val],embeddings=z[val])
    result=dict(state='complete',mode=mode,seed=seed,selected_epoch=selected,completed_epoch=epoch,
        metrics={task:metrics(h,truth) for task,h in outcome.items()},controls=controls,
        checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=digest(folder/'best.pt'),
        frozen_HPO_bitwise_unchanged=True,visual_checkpoint_unchanged=True,test_used=False,
        runtime_seconds=time.monotonic()-started_all,updated_at_utc=now())
    save(folder/'result.json',result);save(folder/'status.json',dict(state='complete',epoch=epoch,updated_at_utc=now()))

if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except Exception as e:
        save(WORK/'cases'/f'{sys.argv[1]}__seed{sys.argv[2]}'/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise
