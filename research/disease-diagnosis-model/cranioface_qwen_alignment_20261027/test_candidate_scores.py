"""Compare cached exhaustive scoring with full teacher forcing on synthetic data."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from candidate_scores import score_candidates,score_candidates_full
from runtime import save,WORK,now
def main():
    torch.set_num_threads(2);torch.manual_seed(13)
    config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True}),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,deepstack_visual_indexes=[]),
        image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
    llm=Qwen3VLForConditionalGeneration(config).eval()
    llm=get_peft_model(llm,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM')).eval()
    prefix=llm.get_input_embeddings()(torch.tensor([[2,3,4,5,6,7]]))
    prefix=prefix.detach().clone();prefix[:,2:4]=torch.randn(1,2,64)*.02
    candidates=[[11,100],[12,13,14,100],[11,15,16,17,100],[100]]
    mean,total=score_candidates(llm,prefix,candidates,0,2)
    direct=[]
    with torch.no_grad():
        for c in candidates:
            embedding=torch.cat((prefix,llm.get_input_embeddings()(torch.tensor([c]))),dim=1)
            n=embedding.shape[1];pos=torch.arange(n).view(1,1,-1).expand(3,1,-1)
            out=llm(inputs_embeds=embedding,attention_mask=torch.ones(1,n,dtype=torch.long),position_ids=pos,use_cache=False)
            logits=out.logits[0,prefix.shape[1]-1:-1]
            direct.append(float(-F.cross_entropy(logits,torch.tensor(c),reduction='sum')))
    np.testing.assert_allclose(total,direct,rtol=1e-6,atol=5e-6)
    np.testing.assert_allclose(mean,np.asarray(direct)/np.asarray(list(map(len,candidates))),rtol=1e-6,atol=5e-6)
    mean_one,total_one=score_candidates(llm,prefix,candidates,0,1)
    np.testing.assert_allclose(total,total_one,rtol=1e-6,atol=5e-6)
    mf,tf=score_candidates_full(llm,prefix,candidates,0,2)
    np.testing.assert_allclose(tf,direct,rtol=1e-6,atol=5e-6)
    np.testing.assert_allclose(mf,mean,rtol=1e-6,atol=5e-6)
    mt,tt=score_candidates_full(llm,prefix,candidates,0,2,include_terminal_input=True)
    np.testing.assert_allclose(tt,direct,rtol=1e-6,atol=5e-6)
    assert len(set(np.argsort(-mean).tolist()))==len(candidates)
    save(WORK/'exhaustive_tests_receipt.json',dict(state='passed',checks=['cached_equals_full_teacher_forcing','full_batch_equals_single_teacher_forcing','variable_length_and_EOS','candidate_batch_invariance','unique_complete_ranking','visual_prompt_embeddings'],updated_at_utc=now()))
    print('Exhaustive likelihood scoring passed synthetic full-forward equivalence tests.')
if __name__=='__main__':main()
