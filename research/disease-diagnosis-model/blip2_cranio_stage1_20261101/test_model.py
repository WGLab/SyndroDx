"""Synthetic-only official architecture and gradient checks."""
import torch
from common import *
from model import Stage1

def main():
    torch.set_num_threads(2);torch.manual_seed(11);model=Stage1().eval()
    tokens=torch.randn(3,8,768);mask=torch.ones(3,8,dtype=torch.long);mask[:,-2:]=0
    ids=torch.tensor([[101,1037,2227,102,0,0],[101,1996,2227,102,0,0],[101,1037,2227,102,0,0]])
    tm=ids.ne(0).long();groups=torch.tensor([0,1,0]);present=torch.tensor([[1,0,0],[0,1,0],[1,0,0]],dtype=torch.bool)
    with torch.no_grad():
        _,qo,a=model.visual(tokens,mask)
        _=model.text(ids,tm)
        _,_,b=model.visual(tokens,mask)
        assert torch.equal(a,b)
        altered=tokens.clone();altered[:,-2:]=1000
        _,_,b=model.visual(altered,mask)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)
        dec=ids.clone();dec[:,0]=model.tokenizer.bos_token_id
        am=torch.cat([torch.ones(3,32,dtype=torch.long),tm],1)
        original=model.Qformer(dec,attention_mask=am,past_key_values=qo.past_key_values,return_dict=True).logits
        dec[:,2]=2048
        changed=model.Qformer(dec,attention_mask=am,past_key_values=qo.past_key_values,return_dict=True).logits
        assert torch.allclose(original[:,:2],changed[:,:2],atol=2e-5,rtol=2e-5)
    allowed=model.allowed_negatives(present)
    assert not allowed[0,2] and allowed[0,1] and not allowed.diag().any()
    model.train();loss,parts=model(tokens,mask,ids,tm,groups,present)
    assert torch.isfinite(loss);loss.backward()
    for name in ['query_tokens','vision_proj.weight','text_proj.weight','itm_head.weight']:
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
    assert model.Qformer.bert.encoder.layer[0].crossattention.self.key.weight.grad is not None
    save(WORK/'tests.json',dict(state='passed',synthetic_only=True,initialization=model.initialization_report,
        parameters=sum(p.numel() for p in model.parameters()),image_queries_independent_of_text=True,
        causal_text_mask_verified=True,visual_padding_mask_verified=True,three_losses_finite=True,
        positive_caption_false_negatives_excluded=True,query_and_crossattention_gradients_verified=True,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'test_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise
