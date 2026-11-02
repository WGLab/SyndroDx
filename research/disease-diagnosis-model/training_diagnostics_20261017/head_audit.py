"""Train-only local DINO-head entropy audit. Never prints patient content."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import gc
import json
import math
import sys
from pathlib import Path

BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1 = BASE.parent / 'dinov3_craniofacial_v1'
WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(V1), str(V1 / 'vendor/dinov3')]
from common import load, save, now, digest, install_rope_dtype_compatibility
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from omegaconf import OmegaConf
from safetensors import safe_open
from PIL import Image
from dinov3.models import build_model_from_cfg
from dinov3.layers.dino_head import DINOHead
from dinov3.data.augmentations import DataAugmentationDINO
from weights import OfficialBackbone


def entropy(p):
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


def probability_summary(probabilities):
    p = probabilities.double()
    conditional = entropy(p).mean()
    marginal = entropy(p.mean(0, keepdim=True))[0]
    return {'mean_sample_entropy': float(conditional), 'marginal_entropy': float(marginal),
            'mutual_information_across_samples': float(marginal - conditional),
            'mean_maximum_probability': float(p.max(-1).values.mean()),
            'top1_prototypes_used': int(p.argmax(-1).unique().numel()),
            'uniform_entropy': math.log(p.shape[-1]),
            'sample_effective_prototypes': float(conditional.exp())}


def heads(kind, dim, cfg):
    opts = cfg.dino
    head = DINOHead(dim, opts.head_n_prototypes, nlayers=opts.head_nlayers,
                    hidden_dim=opts.head_hidden_dim, bottleneck_dim=opts.head_bottleneck_dim)
    if kind in ('warmup200', 'warmup1000'):
        source = WORK / 'warmup_pilot/training/ckpt' / ('199' if kind == 'warmup200' else '999')
    else:
        source = (BASE / 'initialization/teacher_dcp' if kind == 'initial' else
                  BASE / 'training/eval/training_3199/sharded_teacher_checkpoint')
    reader = dcp.FileSystemReader(source)
    metadata = reader.read_metadata()
    prefix = 'model.teacher.dino_head.' if kind in ('warmup200', 'warmup1000') else 'model.dino_head.'
    state = {name: torch.empty(tuple(item.size), dtype=item.properties.dtype)
             for name, item in metadata.state_dict_metadata.items()
             if name.startswith(prefix) and hasattr(item, 'size')}
    if not state:
        raise ValueError('dino_head_not_in_checkpoint')
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    head.load_state_dict({name[len(prefix):]:value for name,value in state.items()}, strict=True)
    return head.cuda().eval()


def backbone(kind, cfg):
    with torch.device('meta'):
        model, _ = build_model_from_cfg(cfg, only_teacher=True)
    model.to_empty(device='cuda')
    model.rope_embed.dtype = torch.float32
    model.init_weights()
    if kind in ('initial', 'warmup200', 'warmup1000'):
        OfficialBackbone(V1 / 'assets/dinov3-vit7b16').load(model)
    else:
        target = model.state_dict()
        with safe_open(BASE / 'encoder/craniofacial_encoder.safetensors', framework='pt', device='cpu') as reader:
            if set(reader.keys()) != set(target):
                raise ValueError('backbone_key_mismatch')
            with torch.no_grad():
                for name, value in target.items():
                    value.copy_(reader.get_tensor(name))
    return model.eval()


@torch.inference_mode()
def main(kind):
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.manual_seed(20261017)
    install_rope_dtype_compatibility()
    destination = BASE / 'diagnostics' / f'head_audit_{kind}.json'
    save(destination, {'state':'running', 'started_at_utc':now(), 'test_used':False,
                       'train_only':True, 'kind':kind})
    folder = BASE / 'validation'
    plan = load(folder / 'plan.json')
    if digest(folder / 'inputs.private.jsonl') != plan['inputs_sha256']:
        raise ValueError('frozen_input_changed')
    with (folder / 'inputs.private.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    selected = []
    seen = set()
    for row in records:
        if row['split'] == 'train' and row['group_id'] not in seen:
            selected.append(row)
            seen.add(row['group_id'])
        if len(selected) == 64:
            break
    if len(selected) < 16:
        raise ValueError('insufficient_training_patients')
    cfg = OmegaConf.load(BASE / 'training/config.yaml')
    model = backbone(kind, cfg)
    head = heads(kind, model.embed_dim, cfg)
    augmentation = DataAugmentationDINO(
        cfg.crops.global_crops_scale, cfg.crops.local_crops_scale, 0,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean, std=cfg.crops.rgb_std)
    logits = []
    for start in range(0, len(selected), 8):
        crops = []
        for row in selected[start:start+8]:
            if digest(row['crop_path']) != row['crop_sha256']:
                raise ValueError('released_crop_changed')
            with Image.open(row['crop_path']) as source:
                image = source.convert('RGB')
                views = augmentation(image)
                crops.extend(views['global_crops_teacher'])
                image.close()
        x = torch.stack(crops).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(x, is_training=True)['x_norm_clstoken']
            output = head(raw).float()
        if not bool(torch.isfinite(output).all()):
            raise ValueError('nonfinite_logits')
        logits.append(output.cpu())
        del x, raw, output
    logits = torch.cat(logits)
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    report = {'state':'complete', 'kind':kind, 'training_patients':len(selected),
              'global_views':len(logits), 'prototype_count':logits.shape[1],
              'train_only':True, 'test_used':False, 'validation_used':False,
              'illustrative_head_diagnostic_not_downstream_evaluation':True,
              'parameters_and_record_content_exported':False, 'temperatures':{},
              'fixed_original_backbone_for_controlled_head_comparison':kind in ('warmup200', 'warmup1000')}
    for temperature in (.04, .07):
        p = torch.softmax(logits / temperature, -1)
        raw_summary = probability_summary(p)
        # Same number of Sinkhorn updates as upstream; one local population,
        # not an exact replay of its distributed training minibatch.
        q = torch.exp(logits.double() / temperature).T
        q /= q.sum()
        k, n = q.shape
        for _ in range(3):
            q /= q.sum(1, keepdim=True)
            q /= k
            q /= q.sum(0, keepdim=True)
            q /= n
        q = (q * n).T
        sk_summary = probability_summary(q)
        report['temperatures'][str(temperature)] = {'softmax':raw_summary, 'sinkhorn':sk_summary}
        del p, q
    report['completed_at_utc'] = now()
    save(destination, report)
    print('Head audit completed; detailed results remain local.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['initial', 'final', 'warmup200', 'warmup1000'], required=True)
    args = parser.parse_args()
    try:
        main(args.kind)
    except Exception as error:
        save(BASE / 'diagnostics' / f'head_audit_{args.kind}.json',
             {'state':'failed', 'error_type':type(error).__name__, 'updated_at_utc':now()})
        raise
