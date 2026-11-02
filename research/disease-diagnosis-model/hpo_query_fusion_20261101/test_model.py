"""Synthetic data only: masking, invariance, gradients and branch preservation."""
import torch
from common import *
from hpo_model import PhenotypeModel
from query_model import QueryFusion

def main():
    torch.set_num_threads(2);torch.manual_seed(11)
    h=PhenotypeModel(torch.randn(20,768),7,9,'hpo')
    m=QueryFusion(h,'conditional_local').eval()
    features=torch.randn(3,512);tokens=torch.randn(3,8,768);mask=torch.ones(3,8,dtype=torch.bool)
    ids=torch.tensor([[1,2,3,0],[3,4,0,0],[0,0,0,0]]);states=torch.zeros_like(ids)
    with torch.no_grad():
        a,z=m(features,tokens,mask,ids,states);base,ref=h(features,ids,states)
        assert torch.equal(a,base) and torch.equal(z,ref)
        m.out.weight.normal_(std=.01)
        a,z=m(features,tokens,mask,ids,states)
        b,w=m(features,tokens[:,torch.arange(7,-1,-1)],mask,ids,states)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)
        b,w=m(features,tokens,mask,ids.flip(1),states.flip(1))
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)
        empty=torch.zeros_like(mask)
        b,w=m(features,tokens,empty,ids,states)
        assert torch.equal(b,base) and torch.equal(w,ref) and torch.isfinite(b).all()
        mask[:,4:]=False
        a,_=m(features,tokens,mask,ids,states)
        altered=tokens.clone();altered[:,4:]=99999
        b,_=m(features,altered,mask,ids,states)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)
    frozen={k:v.clone() for k,v in h.state_dict().items()}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    m.train();opt.zero_grad();a,_=m(features,tokens,mask,ids,states);a.square().mean().backward();opt.step()
    assert all(p.grad is None for p in h.parameters())
    assert all(torch.equal(v,h.state_dict()[k]) for k,v in frozen.items())
    assert m.layers[0].hpo.in_proj_weight.grad is not None
    save(WORK/'tests.json',dict(state='passed',synthetic_only=True,initial_equals_HPO=True,missing_face_exact_HPO_fallback=True,
        HPO_set_permutation_invariance=True,image_token_set_permutation_invariance=True,padding_invariance=True,frozen_HPO_unchanged=True,updated_at_utc=now()))

if __name__=='__main__':main()
