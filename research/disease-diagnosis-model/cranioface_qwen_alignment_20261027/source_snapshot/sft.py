"""Local 32B LoRA disease SFT with an identical HPO-only control arm."""
import datetime,math,os,sys,time
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from runtime import *
from model import Resampler,target_losses
ARM=sys.argv[1];OUT=WORK/'sft'/ARM

def sequences(tok,rows,targets=True,include_visual=True):
    before=tok.encode('<|im_start|>system\nYou are a rare disease research assistant. Use documented evidence to prioritize a diagnosis. Missing findings are unknown. Answer with the disease name only.<|im_end|>\n<|im_start|>user\n'+('Facial evidence:\n' if include_visual else ''),add_special_tokens=False)
    all_ids=[];all_labels=[]
    for r in rows:
        prompt='\n'+r['clinical_input']+'\nWhich diagnosis best matches the evidence?<|im_end|>\n<|im_start|>assistant\n'
        ids=before+([tok.pad_token_id]*32 if include_visual else [])+tok.encode(prompt,add_special_tokens=False)
        target=tok.encode(r['diagnosis']+'<|im_end|>',add_special_tokens=False) if targets else []
        all_labels.append([-100]*len(ids)+target);all_ids.append(ids+target)
    n=max(map(len,all_ids))
    if n>load(WORK/'sft_protocol.json')['max_sequence']:raise RuntimeError('diagnosis_input_exceeds_context_budget')
    mask=[[1]*len(ids)+[0]*(n-len(ids)) for ids in all_ids]
    labels=[ids+[-100]*(n-len(ids)) for ids in all_labels]
    ids=[ids+[tok.pad_token_id]*(n-len(ids)) for ids in all_ids]
    return torch.tensor(ids,device='cuda'),torch.tensor(mask,device='cuda'),torch.tensor(labels,device='cuda'),len(before)

def reduce_grads(parameters,world):
    # Only adapter/connector gradients communicate. The frozen 32B weights are
    # loaded locally and never broadcast at DDP initialization.
    bucket=[];size=0
    def flush(items):
        flat=torch.cat([p.grad.flatten() for p in items]);dist.all_reduce(flat);flat.div_(world);offset=0
        for p in items:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    for p in parameters:
        if p.grad is None:raise RuntimeError('trainable_parameter_missing_gradient')
        bucket.append(p);size+=p.numel()*p.element_size()
        if size>=25*1024*1024:flush(bucket);bucket=[];size=0
    if bucket:flush(bucket)

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);attempt=int(os.environ['CRANIO_ATTEMPT'])
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    p=load(WORK/'sft_protocol.json');torch.manual_seed(p['seed'])
    if digest(WORK/'private/sft_patients.json')!=load(WORK/'sft_data_receipt.json')['patient_sha256']:raise RuntimeError('SFT_input_changed')
    source=load(WORK/'private/sft_patients.json');patients=source['patients'];vocab=source['vocabulary'];classes={r['label'] for r in vocab}
    train=np.asarray([i for i,r in enumerate(patients) if r['split']=='train']);val=np.asarray([i for i,r in enumerate(patients) if r['split']=='validation' and r['label'] in classes])
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');globals=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    for r in patients:sequences(tok,[r])
    if rank==0:save(OUT/'status.json',dict(state='loading_frozen_32B_and_LoRA',world_size=world,updated_at_utc=now()))
    llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
    llm.requires_grad_(False)
    llm=get_peft_model(llm,LoraConfig(r=p['lora_rank'],lora_alpha=p['lora_alpha'],target_modules=p['lora_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});llm.config.use_cache=False
    connector=Resampler().cuda();selected=load(WORK/'best_checkpoint.json')
    if selected['sha256']!=p['initialization_sha256'] or digest(selected['path'])!=selected['sha256']:raise RuntimeError('alignment_initialization_changed')
    connector.load_state_dict(torch.load(selected['path'],map_location='cpu',weights_only=False)['connector'])
    groups=[dict(params=[v for v in llm.parameters() if v.requires_grad],lr=p['lr'])]
    groups.append(dict(params=list(connector.parameters()),lr=p['connector_lr']))
    parameters=[v for group in groups for v in group['params']];opt=torch.optim.AdamW(groups,weight_decay=p['weight_decay'])
    epoch=step=bad=0;best=math.inf
    resume=load(OUT/'latest_checkpoint.json')
    if resume:
        if digest(resume['path'])!=resume['sha256']:raise RuntimeError('SFT_checkpoint_changed')
        state=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=digest(WORK/'sft_protocol.json'):raise RuntimeError('SFT_protocol_changed')
        if ARM=='hpo' and state.get('hpo_control_sha256')!=digest(WORK/'hpo_control_protocol.json'):raise RuntimeError('HPO_control_changed')
        set_peft_model_state_dict(llm,state['lora']);connector.load_state_dict(state['connector']);opt.load_state_dict(state['optimizer'])
        epoch,step,best,bad=(state[k] for k in ['epoch','step','best','bad']);del state
    batch=int(os.environ.get('CRANIO_BATCH',p['per_gpu_batch']));accum=p['accumulation']
    constant=None
    if ARM=='hpo':
        cr=load(WORK/'hpo_control_protocol.json')
        if cr.get('version')!=2 or digest(WORK/'private/hpo_constant_prior.npz')!=cr['prior_sha256']:raise RuntimeError('unsealed_HPO_control')
        with np.load(WORK/'private/hpo_constant_prior.npz') as f:
            constant=(torch.from_numpy(f['patches']).cuda(),torch.from_numpy(f['global_features']).cuda())
    def visual(ids,rng=None):
        if ARM=='hpo':return connector(constant[0].unsqueeze(0).expand(len(ids),-1,-1),constant[1].unsqueeze(0).expand(len(ids),-1))
        pt=[];gl=[]
        for i in ids:
            choices=patients[int(i)]['feature_indices'];ix=rng.choice(choices,2,replace=len(choices)<2).tolist() if rng is not None else (choices[:2] if len(choices)>1 else choices*2)
            pt.append(np.array(patches[ix]).reshape(-1,768));gl.append(np.array(globals[ix]).mean(0))
        return connector(torch.from_numpy(np.stack(pt)).cuda().float(),torch.from_numpy(np.stack(gl)).cuda().float())
    def save_checkpoint(ep):
        path=OUT/'checkpoints'/f'epoch_{ep:03d}.pt';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            torch.save(dict(lora=get_peft_model_state_dict(llm),connector=connector.state_dict(),optimizer=opt.state_dict(),epoch=ep,step=step,best=best,bad=bad,protocol_sha256=digest(WORK/'sft_protocol.json'),hpo_control_sha256=digest(WORK/'hpo_control_protocol.json') if ARM=='hpo' else None),f);f.flush();os.fsync(f.fileno())
        tmp.replace(path);receipt=dict(path=str(path),sha256=digest(path),epoch=ep,step=step,updated_at_utc=now());save(OUT/'latest_checkpoint.json',receipt);return receipt
    @torch.no_grad()
    def evaluate(ep):
        llm.eval();connector.eval();records=[]
        for start in range(0,len(val[rank::world]),batch):
            ids=val[rank::world][start:start+batch];t=sequences(tok,[patients[int(i)] for i in ids])
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=target_losses(llm,visual(ids),t)
            records.extend([[int(i),float(x)] for i,x in zip(ids,loss)])
        gathered=[None]*world;dist.all_gather_object(gathered,records);out=np.array([x for a in gathered for x in a])
        if len(out)!=len(val) or len(set(out[:,0]))!=len(val):raise RuntimeError('validation_coverage_error')
        result=dict(epoch=ep,diagnosis_target_nll=float(out[:,1].mean()),patients=len(val),updated_at_utc=now())
        if rank==0:save(OUT/'latest_validation.json',result);save(OUT/'validation'/f'epoch_{ep:03d}.json',result)
        llm.train();connector.train();return result
    if not resume:
        result=evaluate(0)
        if rank==0:save(OUT/'baseline.json',result);save_checkpoint(0)
        dist.barrier()
    llm.train();connector.train();started=time.monotonic()
    for ep in range(epoch+1,p['max_epochs']+1):
        rng=np.random.default_rng(p['seed']+ep);micro=batch*world;updates=math.ceil(len(train)/(micro*accum));order=np.resize(rng.permutation(train),updates*micro*accum)
        cumulative=0.;n=0
        for u in range(updates):
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                start=(u*accum+a)*micro+rank*batch;ids=order[start:start+batch];t=sequences(tok,[patients[int(i)] for i in ids])
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=target_losses(llm,visual(ids,rng),t).mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite_SFT_loss')
                (loss/accum).backward();cumulative+=float(loss.detach())*len(ids);n+=len(ids)
            reduce_grads(parameters,world);norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite_SFT_gradients')
            opt.step();step+=1;v=torch.tensor([cumulative,n],device='cuda');dist.all_reduce(v)
            if rank==0:save(OUT/'status.json',dict(state='training',arm=ARM,epoch=ep,epoch_update=u+1,epoch_updates=updates,step=step,
                train_nll=float(v[0]/v[1]),gradient_norm=float(norm),world_size=world,per_gpu_batch=batch,effective_batch=micro*accum,
                peak_memory_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),elapsed_seconds=round(time.monotonic()-started),updated_at_utc=now()))
        result=evaluate(ep);improved=result['diagnosis_target_nll']<best-.001
        if improved:best=result['diagnosis_target_nll'];bad=0
        else:bad+=1
        if rank==0:
            ck=save_checkpoint(ep)
            if improved:save(OUT/'best_checkpoint.json',dict(ck,metrics=result))
        dist.barrier()
        if ep>=p['min_epochs'] and bad>=p['patience']:break
        if load(OUT/'resize_requested.json').get('world_size',world)!=world:dist.destroy_process_group();return
    if rank==0:save(OUT/'training_complete.json',dict(state='complete',arm=ARM,epochs=ep,steps=step,best=load(OUT/'best_checkpoint.json'),ranking_evaluation='pending',updated_at_utc=now()))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'workers'/f"{os.environ.get('CRANIO_ATTEMPT','0')}_{os.environ.get('RANK','0')}.failure.json",dict(error_type=type(e).__name__,oom='out of memory' in str(e).lower(),updated_at_utc=now()));raise
