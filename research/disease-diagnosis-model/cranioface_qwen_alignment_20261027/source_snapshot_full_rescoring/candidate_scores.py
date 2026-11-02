"""Exact teacher-forced scoring of every canonical diagnosis, sharing prompt KV."""
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

def fp32_head_forward(self,x):
    # Avoid quantizing the large-vocabulary output logits to BF16 before scoring.
    with torch.autocast(device_type=x.device.type,enabled=False):
        return F.linear(x.float(),self.weight,self.bias)

@torch.no_grad()
def score_candidates_full(llm,prompt_embeddings,candidates,pad_id,batch_size=8):
    """Full teacher forcing for every candidate, with no reused KV state."""
    if prompt_embeddings.shape[0]!=1 or not candidates or any(not c for c in candidates):raise ValueError('invalid_candidates')
    device=prompt_embeddings.device;length=prompt_embeddings.shape[1];sums=[]
    for start in range(0,len(candidates),batch_size):
        batch=candidates[start:start+batch_size];n=len(batch);width=max(map(len,batch))
        ids=torch.full((n,width),pad_id,device=device,dtype=torch.long);valid=torch.zeros_like(ids)
        for i,values in enumerate(batch):
            ids[i,:len(values)]=torch.tensor(values,device=device);valid[i,:len(values)]=1
        embeddings=torch.cat((prompt_embeddings.expand(n,-1,-1),llm.get_input_embeddings()(ids[:,:-1])),dim=1)
        mask=torch.cat((torch.ones(n,length,device=device,dtype=torch.long),valid[:,:-1]),dim=1)
        positions=torch.arange(embeddings.shape[1],device=device).view(1,1,-1).expand(3,n,-1)
        result=llm(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False,logits_to_keep=width)
        logits=result.logits.float()
        nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids.reshape(-1),reduction='none').view(n,width)
        sums.extend((-(nll*valid).sum(1)).cpu().tolist())
        del result,logits,nll,embeddings
    total=torch.tensor(sums,dtype=torch.float64)
    mean=total/torch.tensor(list(map(len,candidates)),dtype=torch.float64)
    if not torch.isfinite(mean).all():raise RuntimeError('nonfinite_full_candidate_scores')
    return mean.numpy(),total.numpy()

@torch.no_grad()
def score_candidates(llm,prompt_embeddings,candidates,pad_id,batch_size=24):
    if prompt_embeddings.shape[0]!=1 or not candidates or any(not c for c in candidates):
        raise ValueError('one_prompt_and_nonempty_candidates_required')
    device=prompt_embeddings.device;length=prompt_embeddings.shape[1]
    positions=torch.arange(length,device=device).view(1,1,-1).expand(3,1,-1)
    pref=llm(inputs_embeds=prompt_embeddings,attention_mask=torch.ones(1,length,device=device,dtype=torch.long),
        position_ids=positions,use_cache=True,logits_to_keep=1)
    cache=pref.past_key_values;original=cache.to_legacy_cache()
    first=F.log_softmax(pref.logits[0,-1].float(),dim=-1);sums=[];start=0
    del pref
    while start<len(candidates):
        batch=candidates[start:start+batch_size];n=len(batch);width=max(map(len,batch))
        ids=torch.full((n,width),pad_id,device=device,dtype=torch.long)
        valid=torch.zeros_like(ids)
        for i,values in enumerate(batch):
            ids[i,:len(values)]=torch.tensor(values,device=device);valid[i,:len(values)]=1
        scores=first[ids[:,0]].clone()
        if width>1:
            # Cache objects are independent; expanded prompt tensors are read-only.
            # DynamicCache appends new keys/values without mutating the shared base.
            work_cache=DynamicCache.from_legacy_cache(tuple((k.expand(n,-1,-1,-1),v.expand(n,-1,-1,-1)) for k,v in original))
            positions=torch.arange(length,length+width-1,device=device).view(1,1,-1).expand(3,n,-1)
            result=llm(input_ids=ids[:,:-1],past_key_values=work_cache,
                attention_mask=torch.cat((torch.ones(n,length,device=device,dtype=torch.long),valid[:,:-1]),dim=1),
                position_ids=positions,cache_position=torch.arange(length,length+width-1,device=device),use_cache=True)
            logits=result.logits.float()
            nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').view(n,-1)
            scores-=(nll*valid[:,1:]).sum(1)
            del result,work_cache,logits,nll
        sums.extend(scores.cpu().tolist());start+=n
        if cache.get_seq_length()!=length:raise RuntimeError('shared_prompt_cache_mutated')
    total=torch.tensor(sums,dtype=torch.float64)
    means=total/torch.tensor([len(x) for x in candidates],dtype=torch.float64)
    if not torch.isfinite(total).all():raise RuntimeError('nonfinite_candidate_score')
    return means.numpy(),total.numpy()
