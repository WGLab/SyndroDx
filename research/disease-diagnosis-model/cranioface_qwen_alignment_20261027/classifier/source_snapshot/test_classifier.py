"""Synthetic tests: direct hidden pooling, padding, gradients, and no vocabulary head."""
import torch
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from classifier_model import classification_head,pooled_hidden
from model import Resampler
from runtime import *
def main():
    torch.set_num_threads(2);torch.manual_seed(4)
    config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True}),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,deepstack_visual_indexes=[]),
        image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
    llm=Qwen3VLForConditionalGeneration(config).requires_grad_(False)
    llm=get_peft_model(llm,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    connector=Resampler(out_dim=64,width=32,heads=4,layers=1,queries=4);head=classification_head(64,7)
    ids=torch.randint(1,80,(2,20));mask=torch.ones_like(ids);mask[0,-4:]=0
    tensors=(ids,mask,torch.full_like(ids,-100),3)
    called=[];handle=llm.get_output_embeddings().register_forward_hook(lambda *args:called.append(True))
    llm.eval();head.eval();visual=connector(torch.randn(2,144,32),torch.randn(2,512))
    a=pooled_hidden(llm,visual,tensors)
    changed=ids.clone();changed[0,-4:]=90
    b=pooled_hidden(llm,visual,(changed,mask,tensors[2],3))
    torch.testing.assert_close(a,b)
    llm.train();head.train();loss=torch.nn.functional.cross_entropy(head(pooled_hidden(llm,visual,tensors)),torch.tensor([1,3]));loss.backward()
    assert connector.query.grad.abs().sum()>0
    assert all(p.grad is not None for p in llm.parameters() if p.requires_grad)
    assert all(p.grad is None for p in llm.parameters() if not p.requires_grad)
    assert not called;handle.remove()
    save(WORK/'classifier/tests.json',dict(state='passed',checks=['input_only_hidden_pooling','padding_invariance','no_vocabulary_head_called','connector_and_LoRA_gradients','frozen_base_weights'],updated_at_utc=now()))
    print('Classifier hidden-state, masking and gradient tests passed.')
if __name__=='__main__':main()
