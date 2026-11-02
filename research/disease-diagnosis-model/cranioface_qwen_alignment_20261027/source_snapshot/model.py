"""Learned visual queries feed a frozen local Qwen language backbone."""
import torch
from torch import nn
import torch.nn.functional as F
class Resampler(nn.Module):
    def __init__(self,out_dim=5120,width=768,heads=8,layers=2,queries=32):
        super().__init__();self.query=nn.Parameter(torch.randn(queries,width)*.02)
        self.global_proj=nn.Linear(512,width);self.memory_norm=nn.LayerNorm(width)
        self.blocks=nn.ModuleList([nn.TransformerDecoderLayer(width,heads,width*4,dropout=0,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,out_dim))
        nn.init.normal_(self.output[-1].weight,std=.001);nn.init.zeros_(self.output[-1].bias)
    def forward(self,patches,global_features):
        memory=torch.cat((patches,self.global_proj(global_features).unsqueeze(1)),dim=1)
        memory=self.memory_norm(memory);q=self.query.unsqueeze(0).expand(patches.shape[0],-1,-1)
        for block in self.blocks:q=block(q,memory)
        return self.output(q)

def build_sequences(tokenizer,texts,query_count,max_length):
    # The HPO answer is present only in the target, never in the input prompt.
    before=tokenizer.encode('<|im_start|>system\nYou are a clinical phenotype research assistant. Describe documented craniofacial findings associated with facial evidence. Unrecorded findings remain unknown.<|im_end|>\n<|im_start|>user\nFacial evidence:\n',add_special_tokens=False)
    after=tokenizer.encode('\nList the recorded craniofacial findings.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    inputs=[];labels=[]
    for text in texts:
        target=tokenizer.encode(text+'<|im_end|>',add_special_tokens=False)
        ids=before+[tokenizer.pad_token_id]*query_count+after+target
        if len(ids)>max_length:raise ValueError('HPO_target_exceeds_frozen_max_sequence')
        inputs.append(ids);labels.append([-100]*(len(before)+query_count+len(after))+target)
    length=max(map(len,inputs));mask=[[1]*len(v)+[0]*(length-len(v)) for v in inputs]
    inputs=[v+[tokenizer.pad_token_id]*(length-len(v)) for v in inputs]
    labels=[v+[-100]*(length-len(v)) for v in labels]
    return torch.tensor(inputs),torch.tensor(mask),torch.tensor(labels),len(before)

def target_losses(llm,visual,tensors):
    ids,mask,labels,offset=tensors
    embeddings=llm.get_input_embeddings()(ids)
    embeddings=torch.cat((embeddings[:,:offset],visual.to(embeddings.dtype),embeddings[:,offset+visual.shape[1]:]),dim=1)
    positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,0)
    result=llm(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions.unsqueeze(0).expand(3,-1,-1),use_cache=False)
    logits=result.logits[:,:-1].float();target=labels[:,1:]
    token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none',ignore_index=-100).view(target.shape)
    counts=(target!=-100).sum(1)
    if not bool((counts>0).all()):raise RuntimeError('target_mask_is_empty')
    return token.sum(1)/counts
