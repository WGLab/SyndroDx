"""All 216 diseases scored for each patient. No beam truncation or duplicates."""
import os,sys,time,types
import numpy as np
import torch
from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from runtime import *
from model import Resampler,target_losses
from sft import sequences
from candidate_scores import score_candidates_full,fp32_head_forward
ARM=sys.argv[1]
def main():
    os.umask(0o077);torch.set_num_threads(4);torch.cuda.set_device(0)
    shard,world=int(sys.argv[2]),int(sys.argv[3]);out=WORK/'ranking_exhaustive'/ARM
    p=load(WORK/'sft_protocol.json');source=load(WORK/'private/sft_patients.json');patients=source['patients'];vocab=source['vocabulary'];classes=[r['label'] for r in vocab]
    val=[i for i,r in enumerate(patients) if r['split']=='validation' and r['label'] in classes][shard::world]
    if digest(WORK/'private/sft_patients.json')!=load(WORK/'sft_data_receipt.json')['patient_sha256']:raise RuntimeError('SFT_patient_input_changed')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    candidates=[tok.encode(r['name']+'<|im_end|>',add_special_tokens=False) for r in vocab]
    if len(set(map(tuple,candidates)))!=len(classes):raise RuntimeError('duplicate_disease_token_sequence')
    llm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
    connector=Resampler().cuda()
    if ARM in ['fusion','hpo']:
        ck=load(WORK/'sft'/ARM/'best_checkpoint.json')
        if digest(ck['path'])!=ck['sha256']:raise RuntimeError('diagnosis_checkpoint_changed')
        state=torch.load(ck['path'],map_location='cpu',weights_only=False)
        llm=get_peft_model(llm,LoraConfig(r=p['lora_rank'],lora_alpha=p['lora_alpha'],target_modules=p['lora_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
        set_peft_model_state_dict(llm,state['lora']);connector.load_state_dict(state['connector']);del state
    elif ARM!='base_hpo':raise RuntimeError('unknown_evaluation_arm')
    llm.requires_grad_(False).eval();connector.requires_grad_(False).eval();llm.config.use_cache=False
    head=llm.get_output_embeddings();head.to(dtype=torch.float32);head.forward=types.MethodType(fp32_head_forward,head)
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');glob=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    if ARM=='hpo':
        if digest(WORK/'private/hpo_constant_prior.npz')!=load(WORK/'hpo_control_protocol.json')['prior_sha256']:raise RuntimeError('control_prior_changed')
        with np.load(WORK/'private/hpo_constant_prior.npz') as f:
            constant=(torch.from_numpy(f['patches']).cuda().unsqueeze(0),torch.from_numpy(f['global_features']).cuda().unsqueeze(0))
    records=[];scores=[];sum_scores=[];check_error=0.;same_shape_error=0.;started=time.monotonic()
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
            batch=8 if ids.shape[1]<=700 else 4 if ids.shape[1]<=1400 else 2
            while True:
                try:
                    mean,total=score_candidates_full(llm,em,candidates,tok.pad_token_id,batch);break
                except torch.cuda.OutOfMemoryError:
                    if batch<=1:raise
                    batch=max(1,batch//2);torch.cuda.empty_cache()
            if not records:
                t=sequences(tok,[row],targets=True,include_visual=ARM!='base_hpo')
                prefix=visual if ARM!='base_hpo' else torch.empty(1,0,5120,device='cuda',dtype=torch.bfloat16)
                direct=float(target_losses(llm,prefix,t)[0]);gold=classes.index(row['label'])
                check_error=abs(direct+mean[gold])
                # Equal-shape reference isolates semantic correctness from BF16
                # GEMM/attention changes caused by candidate batching and padding.
                identical,_=score_candidates_full(llm,em,[candidates[gold]],tok.pad_token_id,1,include_terminal_input=True)
                same_shape_error=abs(direct+identical[0])
                save(out/f'{shard}.numerics.json',dict(batch_vs_single_error=check_error,same_shape_reference_error=same_shape_error,updated_at_utc=now()))
                if same_shape_error>0.0001:raise RuntimeError('identical_shape_reference_scoring_mismatch')
        order=np.argsort(-mean,kind='stable');order_sum=np.argsort(-total,kind='stable')
        if len(order)!=len(classes):raise RuntimeError('incomplete_disease_ranking')
        records.append(dict(index=i,truth=row['label'],ranked=[classes[j] for j in order],ranked_sum=[classes[j] for j in order_sum]))
        scores.append(mean);sum_scores.append(total)
        save(out/f'{shard}.progress.json',dict(completed=len(records),patients=len(val),candidate_diseases=len(classes),
            seconds_per_patient=round((time.monotonic()-started)/len(records),2),peak_memory_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),updated_at_utc=now()))
    private=out/'private';private.mkdir(parents=True,exist_ok=True)
    save(private/f'{shard}.json',dict(records=records,classes=classes))
    np.savez(private/f'{shard}.scores.npz',indices=np.asarray(val),mean_log_probability=np.asarray(scores),sum_log_probability=np.asarray(sum_scores))
    save(out/f'{shard}.receipt.json',dict(state='complete',patients=len(records),world=world,sha256=digest(private/f'{shard}.json'),
        scores_sha256=digest(private/f'{shard}.scores.npz'),candidate_diseases=len(classes),direct_likelihood_check_error=check_error,
        same_shape_reference_error=same_shape_error,method='full_teacher_forcing_fp32_output_head',updated_at_utc=now()))
if __name__=='__main__':main()
