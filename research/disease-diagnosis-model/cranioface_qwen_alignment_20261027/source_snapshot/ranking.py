"""Canonical disease-name beam ranking; all patient-level outputs stay local."""
import os,sys
import numpy as np
import torch
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from runtime import *
from model import Resampler
ARM=sys.argv[1]
# sft imports the arm from argv but does not execute its main function.
from sft import sequences

def trie(vocabulary,tok):
    children={};labels={};eos=tok.convert_tokens_to_ids('<|im_end|>')
    for row in vocabulary:
        ids=tuple(tok.encode(row['name']+'<|im_end|>',add_special_tokens=False))
        if ids[-1]!=eos or eos in ids[:-1]:raise RuntimeError('invalid_canonical_diagnosis_tokens')
        if ids in labels:raise RuntimeError('duplicate_disease_token_sequence')
        labels[ids]=row['label']
        for i in range(len(ids)):children.setdefault(ids[:i],set()).add(ids[i])
    return children,labels,eos

def allowed_tokens(children,eos,prompt_length):
    def allowed(batch_id,sequence):
        prefix=tuple(sequence[prompt_length:].tolist())
        if prefix and prefix[-1]==eos:return [eos]
        if prefix not in children:raise RuntimeError('beam_left_canonical_disease_trie')
        return sorted(children[prefix])
    return allowed

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    shard,world=int(sys.argv[2]),int(sys.argv[3]);out=WORK/'ranking'/ARM
    p=load(WORK/'sft_protocol.json');source=load(WORK/'private/sft_patients.json');patients=source['patients'];vocab=source['vocabulary'];classes={r['label'] for r in vocab}
    val=[i for i,r in enumerate(patients) if r['split']=='validation' and r['label'] in classes][shard::world]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
    connector=Resampler().cuda()
    if ARM in ['fusion','hpo']:
        ck=load(WORK/'sft'/ARM/'best_checkpoint.json')
        if digest(ck['path'])!=ck['sha256']:raise RuntimeError('diagnosis_checkpoint_changed')
        state=torch.load(ck['path'],map_location='cpu',weights_only=False)
        llm=get_peft_model(llm,LoraConfig(r=p['lora_rank'],lora_alpha=p['lora_alpha'],target_modules=p['lora_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
        set_peft_model_state_dict(llm,state['lora']);connector.load_state_dict(state['connector']);del state
    elif ARM!='base_hpo':raise RuntimeError('unknown_evaluation_arm')
    llm.requires_grad_(False).eval();connector.requires_grad_(False).eval();llm.config.use_cache=True
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');glob=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    if ARM=='hpo':
        if digest(WORK/'private/hpo_constant_prior.npz')!=load(WORK/'hpo_control_protocol.json')['prior_sha256']:raise RuntimeError('control_prior_changed')
        with np.load(WORK/'private/hpo_constant_prior.npz') as f:
            constant=(torch.from_numpy(f['patches']).cuda().unsqueeze(0),torch.from_numpy(f['global_features']).cuda().unsqueeze(0))
    children,tokenlabels,eos=trie(vocab,tok);records=[]
    for i in val:
        row=patients[i];ids,mask,labels,offset=sequences(tok,[row],targets=False,include_visual=ARM!='base_hpo')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if ARM=='fusion':
                choices=row['feature_indices'];ix=choices[:2] if len(choices)>1 else choices*2
                pt=torch.from_numpy(np.array(patches[ix]).reshape(1,-1,768)).cuda().float()
                gl=torch.from_numpy(np.array(glob[ix]).mean(0,keepdims=True)).cuda().float();visual=connector(pt,gl)
            elif ARM=='hpo':visual=connector(*constant)
            em=llm.get_input_embeddings()(ids)
            if ARM!='base_hpo':em=torch.cat((em[:,:offset],visual.to(em.dtype),em[:,offset+32:]),dim=1)
            output=llm.generate(input_ids=ids,inputs_embeds=em,attention_mask=mask,max_new_tokens=max(map(len,tokenlabels))+1,
                num_beams=10,num_return_sequences=10,do_sample=False,length_penalty=1.,early_stopping=True,
                use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id,
                prefix_allowed_tokens_fn=allowed_tokens(children,eos,ids.shape[1]),return_dict_in_generate=True)
        ranked=[]
        for sequence in output.sequences:
            suffix=sequence[ids.shape[1]:].tolist()
            if eos not in suffix:raise RuntimeError('incomplete_disease_name_generation')
            key=tuple(suffix[:suffix.index(eos)+1])
            if key not in tokenlabels:raise RuntimeError('invalid_canonical_diagnosis')
            label=tokenlabels[key]
            if label not in ranked:ranked.append(label)
        records.append(dict(index=i,truth=row['label'],ranked=ranked))
        save(out/f'{shard}.progress.json',dict(completed=len(records),patients=len(val),updated_at_utc=now()))
    save(out/'private'/f'{shard}.json',dict(records=records))
    save(out/f'{shard}.receipt.json',dict(state='complete',patients=len(records),world=world,sha256=digest(out/'private'/f'{shard}.json'),updated_at_utc=now()))
if __name__=='__main__':main()
