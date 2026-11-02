"""Exact public ArcFace backbone with an official DINO training-only head."""
import sys
import torch
from torch import nn
import torch.nn.functional as F
from control import WORK, ONNX
sys.path.insert(0, str(WORK/'dependencies'))
from onnx2torch import convert
from vendor.dino_head import DINOHead

def backbone():
    model = convert(str(ONNX))
    # ONNX represents PReLU slopes as constants. Restore only these floating
    # initializer tensors as learnable parameters, preserving every value.
    for node in model.graph.nodes:
        if node.op != 'call_module':
            continue
        module = model.get_submodule(node.target)
        if type(module).__name__ != 'OnnxPReLU':
            continue
        arg = node.args[1]
        if arg.op != 'get_attr':
            raise RuntimeError('unexpected_prelu_initializer')
        parent, name = arg.target.rsplit('.', 1)
        owner = model.get_submodule(parent)
        value = getattr(owner, name)
        if name in owner._buffers:
            del owner._buffers[name]
            owner.register_parameter(name, nn.Parameter(value.clone()))
    return model

def fixed_batchnorm(model):
    # Preserve pretrained population statistics while learning affine weights.
    # Running statistics must not change with world size or clinical pose mix.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

class Encoder(nn.Module):
    def __init__(self, output_dim=65536):
        super().__init__()
        self.backbone = backbone()
        self.head = DINOHead(512, output_dim, use_bn=False, norm_last_layer=True)

    def train(self, mode=True):
        super().train(mode)
        fixed_batchnorm(self.backbone)
        return self

    def forward(self, images):
        features = self.backbone(images)
        if not self.training:
            return self.head(features)
        return self.head(features)

    @torch.no_grad()
    def embed(self, images):
        return F.normalize(self.backbone(images).float(), dim=-1)

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    sp, tp = dict(student.named_parameters()), dict(teacher.named_parameters())
    if sp.keys() != tp.keys():
        raise RuntimeError('teacher_parameter_mismatch')
    for key in sp:
        tp[key].mul_(momentum).add_(sp[key].detach(), alpha=1-momentum)
    sb, tb = dict(student.named_buffers()), dict(teacher.named_buffers())
    if sb.keys() != tb.keys():
        raise RuntimeError('teacher_buffer_mismatch')
    for key in sb:
        tb[key].copy_(sb[key])
