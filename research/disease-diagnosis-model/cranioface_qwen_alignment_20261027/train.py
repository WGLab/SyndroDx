"""Distributed connector-only training against a frozen 32B decoder, offline."""
import contextlib,datetime,math,os,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from runtime import *
from model import Resampler,build_sequences,target_losses

def checkpoint(model,opt,epoch,step,best,bad):
    path=WORK/'checkpoints'/f'epoch_{epoch:03d}.pt';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(connector=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,best=best,bad=bad,
            protocol_sha256=digest(WORK/'protocol.json'),feature_receipt_sha256=digest(WORK/'feature_receipt.json')),f)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    r=dict(path=str(path),sha256=digest(path),epoch=epoch,step=step,updated_at_utc=now());save(WORK/'latest_checkpoint.json',r)
    return r

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);attempt=int(os.environ['CRANIO_ATTEMPT'])
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    p=load(WORK/'protocol.json');torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    r=load(WORK/'feature_receipt.json')
    for path,sha in r['files'].items():
        if digest(WORK/path)!=sha:raise RuntimeError('training_input_changed')
    patients=load(WORK/'private/training_index.json')['patients']
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');global_features=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    train=np.array([i for i,x in enumerate(patients) if x['split']=='train']);val=np.array([i for i,x in enumerate(patients) if x['split']=='validation'])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    # Fail before loading weights if any target would be silently truncated.
    for x in patients:build_sequences(tokenizer,[x['text']],p['queries'],p['max_sequence'])
    save(WORK/'workers'/f'{attempt}_{rank}.json',dict(state='loading_frozen_Qwen_32B',rank=rank,updated_at_utc=now()))
    llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
    llm.requires_grad_(False);llm.config.use_cache=False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    connector=Resampler().cuda();opt=torch.optim.AdamW(connector.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    epoch=step=bad=0;best=math.inf
    resume=load(WORK/'latest_checkpoint.json')
    if resume:
        if digest(resume['path'])!=resume['sha256']:raise RuntimeError('checkpoint_hash_mismatch')
        s=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if s['protocol_sha256']!=digest(WORK/'protocol.json') or s['feature_receipt_sha256']!=digest(WORK/'feature_receipt.json'):raise RuntimeError('resume_inputs_changed')
        connector.load_state_dict(s['connector']);opt.load_state_dict(s['optimizer'])
        epoch,step,best,bad=(s[k] for k in ['epoch','step','best','bad'])
        del s
    model=DDP(connector,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    batch=int(os.environ.get('CRANIO_BATCH',p['per_gpu_batch']));accum=p['accumulation']
    trainable=sum(x.numel() for x in connector.parameters())
    if any(x.requires_grad for x in llm.parameters()):raise RuntimeError('frozen_LLM_has_trainable_weights')
    if rank==0:save(WORK/'status.json',dict(state='loaded',world_size=world,per_gpu_batch=batch,trainable_parameters=trainable,epoch=epoch,step=step,updated_at_utc=now()))
    def features(ids,rng=None):
        pi=[];gi=[]
        for i in ids:
            choices=patients[int(i)]['feature_indices']
            ix=(rng.choice(choices,2,replace=len(choices)<2).tolist() if rng is not None else (choices[:2] if len(choices)>1 else choices*2))
            pi.append(np.array(patches[ix]).reshape(-1,768));gi.append(np.array(global_features[ix]).mean(0))
        return torch.from_numpy(np.stack(pi)).cuda().float(),torch.from_numpy(np.stack(gi)).cuda().float()
    def tensors(ids):
        a,b,c,d=build_sequences(tokenizer,[patients[int(i)]['text'] for i in ids],p['queries'],p['max_sequence'])
        return a.cuda(),b.cuda(),c.cuda(),d
    @torch.no_grad()
    def evaluate(epoch):
        llm.eval();connector.eval();records=[]
        # A fixed non-self patient permutation prevents labels or images crossing splits.
        perm=np.random.default_rng(p['seed']).permutation(len(val));mapping={int(val[perm[j]]):int(val[perm[(j+1)%len(val)]]) for j in range(len(val))}
        local=val[rank::world]
        for start in range(0,len(local),batch):
            ids=local[start:start+batch];pt,gl=features(ids);wrong_pt,wrong_gl=features([mapping[int(i)] for i in ids]);text=tensors(ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                visual=connector(pt,gl);wrong=connector(wrong_pt,wrong_gl)
                paired=target_losses(llm,visual,text);shuffled=target_losses(llm,wrong,text)
                blank=target_losses(llm,torch.zeros_like(visual),text)
            records.extend([[int(i),float(a),float(b),float(c)] for i,a,b,c in zip(ids,paired,shuffled,blank)])
        gathered=[None]*world;dist.all_gather_object(gathered,records)
        out=np.array(sorted([x for part in gathered for x in part]));assert len(out)==len(val) and len(set(out[:,0]))==len(val)
        delta=out[:,1]-out[:,2];rng=np.random.default_rng(p['seed']);boot=np.mean(rng.choice(delta,(2000,len(delta)),replace=True),axis=1)
        summary=dict(epoch=epoch,paired_nll=float(out[:,1].mean()),shuffled_image_nll=float(out[:,2].mean()),blank_image_nll=float(out[:,3].mean()),
            paired_minus_shuffled_nll=float(delta.mean()),paired_minus_shuffled_nll_ci95=np.quantile(boot,[.025,.975]).tolist(),
            validation_patients=len(val),updated_at_utc=now())
        if rank==0:
            save(WORK/'validation'/f'epoch_{epoch:03d}.json',summary);save(WORK/'latest_validation.json',summary)
            np.save(WORK/'private'/f'validation_{epoch:03d}.npy',out)
        connector.train();llm.train()
        return summary
    dist.barrier()
    if not resume:
        first=evaluate(0)
        if rank==0:
            save(WORK/'baseline.json',first);checkpoint(connector,opt,0,0,best,bad)
        dist.barrier()
    connector.train();llm.train();started=time.monotonic()
    for ep in range(epoch+1,p['max_epochs']+1):
        rng=np.random.default_rng(p['seed']+ep);order=rng.permutation(train)
        micro=batch*world;steps=math.ceil(len(order)/(micro*accum));needed=steps*micro*accum
        order=np.resize(order,needed)
        running=0.;observations=0
        for update in range(steps):
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                start=(update*accum+a)*micro+rank*batch;ids=order[start:start+batch]
                pt,gl=features(ids,rng);t=tensors(ids)
                with (model.no_sync() if a<accum-1 else contextlib.nullcontext()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=target_losses(llm,model(pt,gl),t).mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite_alignment_loss')
                    (loss/accum).backward()
                running+=float(loss.detach())*len(ids);observations+=len(ids)
            grad=torch.nn.utils.clip_grad_norm_(connector.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite_connector_gradient')
            opt.step();step+=1
            vals=torch.tensor([running,observations],device='cuda');dist.all_reduce(vals)
            if rank==0:
                save(WORK/'status.json',dict(state='training',phase='frozen_32B_visual_connector',epoch=ep,epoch_update=update+1,epoch_updates=steps,
                    step=step,train_nll=float(vals[0]/vals[1]),world_size=world,per_gpu_batch=batch,effective_batch=micro*accum,
                    connector_gradient_norm=float(grad),peak_memory_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),elapsed_seconds=round(time.monotonic()-started),updated_at_utc=now()))
        result=evaluate(ep);improved=result['paired_nll']<best-0.001
        if improved:best=result['paired_nll'];bad=0
        else:bad+=1
        if rank==0:
            ck=checkpoint(connector,opt,ep,step,best,bad)
            if improved:save(WORK/'best_checkpoint.json',dict(ck,metrics=result))
        dist.barrier()
        if ep>=p['min_epochs'] and bad>=p['patience']:break
        resize=load(WORK/'resize_requested.json').get('world_size',world)
        if resize!=world:
            dist.destroy_process_group();return
    if rank==0:
        selected=load(WORK/'best_checkpoint.json');metrics=selected['metrics'];baseline=load(WORK/'baseline.json')
        passed=metrics['paired_minus_shuffled_nll_ci95'][1]<0 and metrics['paired_nll']<baseline['paired_nll']
        save(WORK/'training_complete.json',dict(state='complete',best_checkpoint=selected,epochs=ep,steps=step,
            visual_conditioning_gate=bool(passed),clinical_diagnostic_benefit_established=False,
            followon='paired disease SFT and clinical evaluation require their own controlled protocol; no fabricated rationales',updated_at_utc=now()))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'workers'/f"{os.environ.get('CRANIO_ATTEMPT','0')}_{os.environ.get('RANK','0')}.failure.json",dict(error_type=type(e).__name__,oom='out of memory' in str(e).lower(),updated_at_utc=now()))
        raise
