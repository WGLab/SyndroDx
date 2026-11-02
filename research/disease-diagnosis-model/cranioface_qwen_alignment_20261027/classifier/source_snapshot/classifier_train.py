"""Frozen representation probe, followed by matched Qwen classifier LoRA training."""
import datetime,math,os,sys,time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from runtime import *
from model import Resampler
from sft import sequences,reduce_grads
from classifier_model import classification_head,pooled_hidden
ARM=sys.argv[1];OUT=WORK/'classifier'/ARM
def main():
    os.umask(0o077);(OUT/'private').mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.cuda.set_device(0)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);attempt=int(os.environ['CRANIO_ATTEMPT'])
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    p=load(WORK/'classifier/protocol.json');torch.manual_seed(p['seed'])
    if digest(WORK/'private/sft_patients.json')!=p['patient_sha256'] or digest(WORK/'feature_receipt.json')!=p['feature_receipt_sha256']:raise RuntimeError('classifier_inputs_changed')
    data=load(WORK/'private/sft_patients.json');patients=data['patients'];classes=[r['label'] for r in data['vocabulary']];mapping={x:i for i,x in enumerate(classes)}
    train=np.asarray([i for i,r in enumerate(patients) if r['split']=='train']);val=np.asarray([i for i,r in enumerate(patients) if r['split']=='validation' and r['label'] in mapping])
    labels=torch.tensor([mapping.get(r['label'],-1) for r in patients],device='cuda')
    counts=torch.bincount(labels[train],minlength=len(classes)).float()
    if not bool((counts>0).all()):raise RuntimeError('missing_training_class')
    weights=counts.rsqrt();weights/=weights.mean()
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');glob=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    for r in patients:sequences(tok,[r],targets=False)
    if rank==0:save(OUT/'status.json',dict(state='loading_Qwen_32B',world_size=world,updated_at_utc=now()))
    llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True).requires_grad_(False)
    llm=get_peft_model(llm,LoraConfig(r=p['lora_rank'],lora_alpha=p['lora_alpha'],target_modules=p['lora_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});llm.config.use_cache=False
    connector=Resampler().cuda();ck=p['aligned_connector']
    if digest(ck['path'])!=ck['sha256']:raise RuntimeError('connector_initialization_changed')
    connector.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['connector'])
    head=classification_head(classes=len(classes)).cuda()
    # Explicit synchronization of all trained initial values across independent leases.
    for v in [v for v in llm.parameters() if v.requires_grad]+list(connector.parameters())+list(head.parameters()):dist.broadcast(v.data,src=0)
    if ARM=='hpo':
        if digest(WORK/'hpo_control_protocol.json')!=p['HPO_control_sha256']:raise RuntimeError('control_changed')
        with np.load(WORK/'private/hpo_constant_prior.npz') as f:constant=(torch.from_numpy(f['patches']).cuda(),torch.from_numpy(f['global_features']).cuda())
    def hidden(ids,rng=None):
        if ARM=='hpo':pt=constant[0].unsqueeze(0).expand(len(ids),-1,-1);gl=constant[1].unsqueeze(0).expand(len(ids),-1)
        else:
            a=[];b=[]
            for i in ids:
                choices=patients[int(i)]['feature_indices'];ix=rng.choice(choices,2,replace=len(choices)<2).tolist() if rng is not None else (choices[:2] if len(choices)>1 else choices*2)
                a.append(np.array(patches[ix]).reshape(-1,768));b.append(np.array(glob[ix]).mean(0))
            pt=torch.from_numpy(np.stack(a)).cuda().float();gl=torch.from_numpy(np.stack(b)).cuda().float()
        return pooled_hidden(llm,connector(pt,gl),sequences(tok,[patients[int(i)] for i in ids],targets=False))
    def gather(rows):
        all_rows=[None]*world;dist.all_gather_object(all_rows,rows)
        return sorted([x for group in all_rows for x in group],key=lambda x:x[0])
    def metrics(rows):
        if len(rows)!=len(val) or [r[0] for r in rows]!=val.tolist():raise RuntimeError('validation_patient_coverage_error')
        scores=np.stack([r[1] for r in rows]);truth=labels[val].cpu().numpy();order=np.argsort(-scores,axis=1,kind='stable')
        result={f'macro_top{k}':float(np.mean([np.mean(np.any(order[truth==c,:k]==c,axis=1)) for c in np.unique(truth)])) for k in [1,5,10]}
        result['loss']=float(F.cross_entropy(torch.from_numpy(scores),torch.from_numpy(truth)).item());result['patients']=len(rows)
        return result,scores
    def tensors_for_step(order,start,total):
        batch=order[start:start+total];local=batch[rank::world]
        width=math.ceil(total/world);mask=np.arange(width)<len(local)
        local=np.pad(local,(0,width-len(local)),constant_values=int(batch[0]))
        return local,torch.tensor(mask,device='cuda',dtype=torch.float32)
    def train_head_probe(features):
        opt=torch.optim.AdamW(head.parameters(),lr=p['head_probe_lr'],weight_decay=.01);best=(-1.,-1.);state=None;report=None;best_scores=None
        for ep in range(1,p['head_probe_epochs']+1):
            head.train();rng=np.random.default_rng(p['seed']+ep);total=p['head_probe_global_batch'];order=np.resize(rng.permutation(train),math.ceil(len(train)/total)*total)
            for start in range(0,len(order),total):
                ids,valid=tensors_for_step(order,start,total);opt.zero_grad(set_to_none=True)
                denominator=weights[labels[order[start:start+total]]].sum()
                loss=(F.cross_entropy(head(features[ids]),labels[ids],weight=weights,reduction='none')*valid).sum()*world/denominator
                loss.backward();reduce_grads(list(head.parameters()),world);opt.step()
            head.eval()
            with torch.no_grad():scores=head(features[val]).float().cpu().numpy()
            report,current=metrics([(int(i),s) for i,s in zip(val,scores)]);key=(report['macro_top5'],report['macro_top1'])
            if key>best:best=key;state={k:v.detach().clone() for k,v in head.state_dict().items()};best_scores=current.copy();best_report=dict(report,epoch=ep)
        head.load_state_dict(state)
        if rank==0:
            np.save(OUT/'private/probe_scores.npy',best_scores);torch.save(head.state_dict(),OUT/'probe_head.pt')
            save(OUT/'probe_result.json',dict(state='complete',best=best_report,selection='development macro Top5, Top1 tie-break',updated_at_utc=now()))
    epoch=step=bad=0;best=(-1.,-1.)
    groups=[dict(params=[v for v in llm.parameters() if v.requires_grad],lr=p['lora_lr']),dict(params=list(connector.parameters()),lr=p['connector_lr']),dict(params=list(head.parameters()),lr=p['head_lr'])]
    parameters=[v for g in groups for v in g['params']];opt=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3)
    resume=load(OUT/'latest_checkpoint.json')
    if resume:
        if digest(resume['path'])!=resume['sha256']:raise RuntimeError('classifier_checkpoint_changed')
        s=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if s['protocol_sha256']!=digest(WORK/'classifier/protocol.json'):raise RuntimeError('classifier_protocol_changed')
        set_peft_model_state_dict(llm,s['lora']);connector.load_state_dict(s['connector']);head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        epoch,step,bad=s['epoch'],s['step'],s['bad'];best=tuple(s['best']);del s
    else:
        llm.eval();connector.eval();rows=[];all_ids=np.concatenate((train,val))
        with torch.no_grad():
            for start in range(0,len(all_ids[rank::world]),2):
                ids=all_ids[rank::world][start:start+2]
                with torch.autocast('cuda',dtype=torch.bfloat16):values=hidden(ids)
                if not torch.isfinite(values).all():raise RuntimeError('nonfinite_probe_features')
                rows.extend([(int(i),v.cpu().numpy()) for i,v in zip(ids,values)])
                if rank==0:save(OUT/'status.json',dict(state='extracting_frozen_probe_features',completed=start+len(ids),rank_patients=len(all_ids[rank::world]),world_size=world,updated_at_utc=now()))
        rows=gather(rows)
        if len(rows)!=len(all_ids) or len({r[0] for r in rows})!=len(rows):raise RuntimeError('probe_feature_coverage_error')
        features=torch.zeros(len(patients),5120,device='cuda')
        for i,v in rows:features[i]=torch.from_numpy(v).cuda()
        train_head_probe(features);del features,rows
    @torch.no_grad()
    def evaluate(ep):
        llm.eval();connector.eval();head.eval();rows=[]
        for start in range(0,len(val[rank::world]),2):
            ids=val[rank::world][start:start+2]
            with torch.autocast('cuda',dtype=torch.bfloat16):scores=head(hidden(ids))
            rows.extend([(int(i),s.float().cpu().numpy()) for i,s in zip(ids,scores)])
        result,scores=metrics(gather(rows));result.update(epoch=ep,step=step,updated_at_utc=now())
        if rank==0:save(OUT/'validation'/f'{ep:03d}.json',result);save(OUT/'latest_validation.json',result)
        llm.train();connector.train();head.train();return result,scores
    def checkpoint(ep):
        path=OUT/'checkpoints'/f'epoch_{ep:03d}.pt';path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:
            torch.save(dict(lora=get_peft_model_state_dict(llm),connector=connector.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=ep,step=step,bad=bad,best=best,protocol_sha256=digest(WORK/'classifier/protocol.json')),f);f.flush();os.fsync(f.fileno())
        temp.replace(path);r=dict(path=str(path),sha256=digest(path),epoch=ep,step=step);save(OUT/'latest_checkpoint.json',r);return r
    if not resume:
        result,scores=evaluate(0);best=(result['macro_top5'],result['macro_top1'])
        if rank==0:
            r=checkpoint(0);np.save(OUT/'private/best_scores.npy',scores);save(OUT/'best_checkpoint.json',dict(r,metrics=result))
        dist.barrier()
    micro=int(os.environ.get('CRANIO_BATCH',p['per_gpu_batch']));started=time.monotonic()
    for ep in range(epoch+1,p['joint_max_epochs']+1):
        llm.train();connector.train();head.train();rng=np.random.default_rng(p['seed']+ep);total=p['joint_global_batch'];updates=math.ceil(len(train)/total);order=np.resize(rng.permutation(train),updates*total);cumulative=0.
        for u in range(updates):
            ids,valid=tensors_for_step(order,u*total,total);opt.zero_grad(set_to_none=True)
            for start in range(0,len(ids),micro):
                local=ids[start:start+micro];mask=valid[start:start+micro]
                denominator=weights[labels[order[u*total:(u+1)*total]]].sum()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=(F.cross_entropy(head(hidden(local,rng)),labels[local],weight=weights,reduction='none')*mask).sum()*world/denominator
                if not torch.isfinite(loss):raise RuntimeError('nonfinite_classifier_loss')
                loss.backward();cumulative+=float(loss.detach())
            reduce_grads(parameters,world);norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite_classifier_gradient')
            opt.step();step+=1;v=torch.tensor(cumulative,device='cuda');dist.all_reduce(v)
            if rank==0:save(OUT/'status.json',dict(state='joint_classifier_training',epoch=ep,epoch_update=u+1,epoch_updates=updates,step=step,train_weighted_CE=float(v/world/(u+1)),gradient_norm=float(norm),world_size=world,global_batch=total,per_gpu_microbatch=micro,peak_memory_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),elapsed_seconds=round(time.monotonic()-started),updated_at_utc=now()))
        result,scores=evaluate(ep);scheduler.step(result['loss']);key=(result['macro_top5'],result['macro_top1']);improved=key>best
        if improved:best=key;bad=0
        else:bad+=1
        if rank==0:
            r=checkpoint(ep)
            if improved:np.save(OUT/'private/best_scores.npy',scores);save(OUT/'best_checkpoint.json',dict(r,metrics=result))
        dist.barrier()
        if ep>=p['joint_min_epochs'] and bad>=p['joint_patience']:break
        if load(OUT/'resize_requested.json').get('world_size',world)!=world:dist.destroy_process_group();return
    if rank==0:save(OUT/'result.json',dict(state='complete',best=load(OUT/'best_checkpoint.json'),probe=load(OUT/'probe_result.json'),epochs=ep,stop_reason='development_Top5_patience' if bad>=p['joint_patience'] else 'configured_epoch_budget',loss_convergence_not_assumed=True,updated_at_utc=now()))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'workers'/f"{os.environ.get('CRANIO_ATTEMPT','0')}_{os.environ.get('RANK','0')}.failure.json",dict(error_type=type(e).__name__,oom='out of memory' in str(e).lower(),updated_at_utc=now()));raise
