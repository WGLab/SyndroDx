"""Synthetic test: Qwen VL text path, visual-prefix gradients, and LoRA isolation."""
import torch
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from model import Resampler,target_losses
from runtime import save,WORK,now
def main():
    torch.set_num_threads(2);torch.manual_seed(1)
    config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True}),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,deepstack_visual_indexes=[]),
        image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
    llm=Qwen3VLForConditionalGeneration(config).requires_grad_(False)
    llm=get_peft_model(llm,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});llm.train()
    connector=Resampler(out_dim=64,width=32,heads=4,layers=1,queries=4)
    with torch.no_grad():
        constant=connector(torch.ones(2,144,32),torch.ones(2,512))
        assert torch.allclose(constant[0],constant[1])
    ids=torch.randint(1,80,(2,20));mask=torch.ones_like(ids);labels=ids.clone();labels[:,:10]=-100
    visual=connector(torch.randn(2,144,32),torch.randn(2,512));loss=target_losses(llm,visual,(ids,mask,labels,3)).mean();loss.backward()
    assert torch.isfinite(loss) and connector.query.grad.abs().sum()>0
    trainable=[(n,p) for n,p in llm.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and p.grad is not None for n,p in trainable)
    assert all(p.grad is None for p in llm.parameters() if not p.requires_grad)
    state=get_peft_model_state_dict(llm);set_peft_model_state_dict(llm,state)
    # Exercise the exact inputs_embeds + input_ids beam interface on a tiny model.
    import sys
    sys.argv=['test','fusion']
    from ranking import trie,allowed_tokens
    class TinyTokenizer:
        def encode(self,text,add_special_tokens=False):return [ord(c)%70+1 for c in text.removesuffix('<|im_end|>')]+[100]
        def convert_tokens_to_ids(self,text):return 100
    vocabulary=[dict(label=str(i),name=chr(65+i)) for i in range(12)]
    children,labels_by_tokens,eos=trie(vocabulary,TinyTokenizer());llm.eval()
    prompt=ids[:1,:10];em=llm.get_input_embeddings()(prompt).detach();em[:,3:7]=visual[:1].detach()
    with torch.no_grad():
        result=llm.generate(input_ids=prompt,inputs_embeds=em,attention_mask=torch.ones_like(prompt),
            max_new_tokens=3,num_beams=10,num_return_sequences=10,do_sample=False,
            prefix_allowed_tokens_fn=allowed_tokens(children,eos,prompt.shape[1]),
            eos_token_id=eos,pad_token_id=0,use_cache=True,early_stopping=True)
    assert result.shape[0]==10
    for row in result:
        suffix=row[prompt.shape[1]:].tolist();assert eos in suffix
        assert tuple(suffix[:suffix.index(eos)+1]) in labels_by_tokens
    save(WORK/'ranking_tests_receipt.json',dict(state='passed',checks=['canonical_disease_trie','Qwen_LoRA_generate_inputs_embeds','10_beam_canonical_rankings'],updated_at_utc=now()))
    save(WORK/'sft_tests_receipt.json',dict(state='passed',checks=['Qwen_VL_inputs_embeds','gradient_checkpointing','prefix_conditioning_gradient','only_LoRA_weights_trainable','adapter_state_roundtrip'],updated_at_utc=now()))
    print('Synthetic Qwen/LoRA gradient and checkpoint tests passed.')
if __name__=='__main__':main()
