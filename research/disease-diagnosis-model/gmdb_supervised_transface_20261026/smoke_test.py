"""Synthetic-only CPU test of actual weights, gradients and adaptation restore."""
import os
import torch
from common import WORK,load,save,now
from data import verify_data
from disease_model import DiseaseModel

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.manual_seed(5)
    verify_data()
    selection=load(WORK/'smoke/identity_checkpoint.json')
    results={}
    for arm in ('original','identity_cpt'):
        model=DiseaseModel(7,arm,selection);model.train()
        inputs=torch.randn(2,3,112,112);labels=torch.tensor([1,3])
        old=model.classifier[1].weight.detach().clone()
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
        logits,_=model(inputs);loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward()
        trainable=[p for p in model.parameters() if p.requires_grad]
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert all(p.grad is None for p in model.face.blocks.parameters())
        assert not model.face.training and model.face.feature.training
        optimizer.step();assert not torch.equal(old,model.classifier[1].weight)
        model.eval()
        with torch.no_grad():expected=model(inputs)[0]
        path=WORK/'smoke'/f'{arm}.pt';torch.save(model.adaptation_state(),path)
        with torch.no_grad():model.classifier[1].weight.zero_()
        model.load_adaptation(torch.load(path,weights_only=True))
        with torch.no_grad():torch.testing.assert_close(model(inputs)[0],expected)
        results[arm]=dict(finite_gradients=True,frozen_transformer=True,adaptation_restore=True)
        path.unlink();del model,optimizer,trainable,logits,loss
    save(WORK/'smoke/receipt.json',dict(state='passed',synthetic_only=True,models=results,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'smoke/receipt.json',dict(state='failed',error_type=type(e).__name__,updated_at_utc=now()));raise
