"""Inference-only networks matching the locked seed11 release.

Visual layout follows TransFace-L (https://github.com/DanJun6737/TransFace).
This implementation has no training initialization, downloads or timm dependency.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=9, stride=9)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 2304, bias=False)
        self.proj = nn.Linear(768, 768)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, 8, 96).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0., scale=96 ** -.5)
        return self.proj(x.transpose(1, 2).reshape(b, t, w))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(F.relu6(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(768)
        self.mlp = Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 144, 768))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.blocks = nn.ModuleList([Block() for _ in range(24)])
        self.norm = nn.LayerNorm(768)
        self.senet = nn.Sequential(nn.Linear(768 * 144, 144, bias=False), nn.ReLU(),
                                   nn.Linear(144, 144, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.float())
        return (x * self.senet(x.flatten(1)).unsqueeze(-1)).flatten(1)


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = Body()
        self.feature = nn.Sequential(nn.Linear(768 * 144, 768, bias=False),
            nn.BatchNorm1d(768, eps=2e-5), nn.Linear(768, 512, bias=False),
            nn.BatchNorm1d(512, eps=2e-5))

    def forward(self, x):
        return F.normalize(self.feature(self.body(x)).float(), dim=-1)


class PhenotypeModel(nn.Module):
    def __init__(self, embeddings, classes=216, public_classes=612, mode='hpo', width=256):
        super().__init__()
        self.mode = mode
        self.terms = nn.Embedding.from_pretrained(embeddings.float(), freeze=True, padding_idx=0)
        self.project = nn.Sequential(nn.Linear(embeddings.shape[1], width), nn.LayerNorm(width), nn.GELU())
        self.assertion = nn.Embedding(3, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, .1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.visual = nn.Sequential(nn.Linear(512, width), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, 8, dropout=.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
        self.fused_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(width, classes))
        self.public_head = nn.Linear(width, public_classes)

    def represent(self, features, ids, states):
        z = self.project(self.terms(ids)) + self.assertion(states)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], dim=1)
        valid = torch.cat([torch.ones(len(ids), 1, dtype=torch.bool, device=ids.device), ids.ne(0)], dim=1)
        hidden = self.encoder(z, src_key_padding_mask=~valid)
        h = hidden[:, 0]
        if self.mode == 'hpo':
            return h
        image = self.visual(features.float())
        evidence, _ = self.attention(image[:, None], hidden, hidden, key_padding_mask=~valid, need_weights=False)
        gate = self.gate(torch.cat([h, image], dim=1))
        available = features.float().norm(dim=1, keepdim=True).gt(0).float()
        return self.fused_norm(h + available * gate * (image + evidence[:, 0]))

    def forward(self, features, ids, states):
        z = self.represent(features, ids, states)
        return self.head(z), z
