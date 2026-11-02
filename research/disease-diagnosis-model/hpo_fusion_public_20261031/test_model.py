"""Synthetic checks for modality isolation, set semantics and missing findings."""
import json
import numpy as np
import torch
from common import WORK, save, now
from model import PhenotypeModel, profile_loss
from prepare import pack


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    table = torch.randn(40, 32)
    table[0] = 0
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
    states = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
    features = torch.randn(3, 512)
    hpo = PhenotypeModel(table, 4, 6, 'hpo', width=256).eval()
    a, _ = hpo(features, ids, states)
    b, _ = hpo(features * 23, ids, states)
    assert torch.equal(a, b), 'HPO-only model depends on image'
    permutation = torch.tensor([2, 0, 3, 1])
    c, _ = hpo(features, ids[:, permutation], states[:, permutation])
    assert torch.allclose(a, c, atol=2e-6), 'HPO ordering changes the set representation'
    for mode in ['hpo', 'fusion_selected', 'fusion_raw']:
        model = PhenotypeModel(table, 4, 6, mode)
        for mask in [ids, torch.zeros_like(ids)]:
            logits, z = model(torch.zeros_like(features), mask, states)
            assert torch.isfinite(logits).all() and torch.isfinite(z).all()
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
            loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        assert model.terms.weight.grad is None, 'Frozen biomedical term embeddings changed'
    try:
        pack(['HP:0000001'], ['HP:0000001'], {'HP:0000001': 1})
        raise RuntimeError('Contradictory annotations admitted')
    except AssertionError:
        pass
    ii, ss, _ = pack(['HP:0000001'], [], {'HP:0000001': 1, 'HP:0000002': 2})
    assert ii[0] == 1 and not (ii == 2).any() and not ss.any(), 'Missing phenotype became a negative'
    save(WORK / 'tests.json', dict(state='passed', checks=[
        'HPO-only image independence', 'HPO set permutation invariance',
        'finite missing-image and missing-HPO gradients in all modes',
        'frozen biomedical embeddings', 'contradictory assertions rejected',
        'unrecorded phenotype not converted to negative'], updated_at_utc=now()))
    print(json.dumps(dict(state='passed', checks=6)))


if __name__ == '__main__':
    main()
