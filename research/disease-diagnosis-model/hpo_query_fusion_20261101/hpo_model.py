"""A set-valued HPO encoder and a frozen-feature facial fusion branch."""
import torch
from torch import nn
import torch.nn.functional as F


class PhenotypeModel(nn.Module):
    def __init__(self, embeddings, classes, public_classes, mode, width=256):
        super().__init__()
        self.mode = mode
        self.terms = nn.Embedding.from_pretrained(embeddings.float(), freeze=True, padding_idx=0)
        self.project = nn.Sequential(nn.Linear(embeddings.shape[1], width), nn.LayerNorm(width), nn.GELU())
        self.assertion = nn.Embedding(2, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.cls, std=.02)
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, .1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.visual = nn.Sequential(nn.Linear(512, width), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, 8, dropout=.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
        self.fused_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(width, classes))
        self.public_head = nn.Linear(width, public_classes)

    def phenotype(self, ids, states):
        z = self.project(self.terms(ids)) + self.assertion(states)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], dim=1)
        valid = torch.cat([torch.ones(len(ids), 1, dtype=torch.bool, device=ids.device), ids.ne(0)], dim=1)
        hidden = self.encoder(z, src_key_padding_mask=~valid)
        return hidden[:, 0], hidden, valid

    def represent(self, features, ids, states):
        h, tokens, mask = self.phenotype(ids, states)
        if self.mode == 'hpo':
            return h
        image = self.visual(features.float())
        evidence, _ = self.attention(image[:, None], tokens, tokens, key_padding_mask=~mask, need_weights=False)
        gate = self.gate(torch.cat([h, image], dim=1))
        # Missing-image observations must use the text path without a learned fake image.
        available = features.float().norm(dim=1, keepdim=True).gt(0).float()
        return self.fused_norm(h + available * gate * (image + evidence[:, 0]))

    def forward(self, features, ids, states):
        z = self.represent(features, ids, states)
        return self.head(z), z


def profile_loss(model, representation, labels, profiles, offset=0):
    available = profiles['available'][labels + offset]
    if available.sum() < 2:
        return representation.sum() * 0
    labels = labels[available]
    unique, inverse = labels.unique(sorted=True, return_inverse=True)
    if len(unique) < 2:
        return representation.sum() * 0
    p, _, _ = model.phenotype(profiles['ids'][unique + offset], profiles['states'][unique + offset])
    scores = F.normalize(representation[available].float(), dim=-1) @ F.normalize(p.float(), dim=-1).T / .07
    return F.cross_entropy(scores, inverse)
