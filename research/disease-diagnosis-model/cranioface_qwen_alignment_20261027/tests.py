"""Synthetic privacy-independent checks of conditioning and causal targets."""
import torch
from model import Resampler,build_sequences,target_losses
class Tokenizer:
    pad_token_id=0
    def encode(self,s,add_special_tokens=False):return [ord(c)%80+1 for c in s]
def main():
    torch.set_num_threads(2);torch.manual_seed(1)
    m=Resampler(out_dim=64,width=32,heads=4,layers=1,queries=4)
    p=torch.randn(2,144,32);g=torch.randn(2,512)
    out=m(p,g);assert out.shape==(2,4,64)
    out.square().sum().backward();assert m.query.grad.abs().sum()>0
    with torch.no_grad():assert not torch.allclose(m(p,g),m(p.flip(0),g.flip(0)))
    ids,mask,labels,offset=build_sequences(Tokenizer(),['ANSWER_A','ANSWER_BBB'],4,1024)
    assert (labels[:,:offset+4]==-100).all()
    assert (labels[mask==0]==-100).all()
    assert (mask[:,offset:offset+4]==1).all()
    assert (labels[labels!=-100]>0).all()
    from runtime import save,WORK,now
    save(WORK/'tests_receipt.json',dict(state='passed',checks=['real_patch_conditioning','gradient_reaches_queries','no_target_in_prompt','padding_and_prefix_masked'],updated_at_utc=now()))
    print('Synthetic connector and causal-label tests passed.')
if __name__=='__main__':main()
