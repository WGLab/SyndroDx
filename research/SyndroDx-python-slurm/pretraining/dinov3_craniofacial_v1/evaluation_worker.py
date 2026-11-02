"""Offline frozen-feature extraction. No image, embedding, or record leaves HPC."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from torchvision.transforms.functional import pil_to_tensor
from runtime import ROOT, load, save, digest, now
from weights import OfficialBackbone
from dinov3.models import build_model_from_cfg


def encoder(kind):
    cfg = OmegaConf.load(ROOT / 'encoder/training_config.yaml')
    with torch.device('meta'):
        model, _ = build_model_from_cfg(cfg, only_teacher=True)
    model.to_empty(device='cuda')
    model.rope_embed.dtype = torch.float32
    model.init_weights()  # Initializes the RoPE and QKV mask buffers too.
    if kind == 'original_dinov3':
        OfficialBackbone(ROOT / 'assets/dinov3-vit7b16').load(model)
    else:
        targets = model.state_dict()
        with safe_open(ROOT / 'encoder/craniofacial_encoder.safetensors', framework='pt', device='cpu') as weights:
            if set(weights.keys()) != set(targets):
                raise RuntimeError('exported_encoder_key_mismatch')
            with torch.no_grad():
                for name, target in targets.items():
                    value = weights.get_tensor(name)
                    if value.shape != target.shape:
                        raise RuntimeError('exported_encoder_shape_mismatch')
                    target.copy_(value)
                    del value
    return model.eval(), cfg


def fit_image(path, cfg):
    size = cfg.crops.global_crops_size
    mean = tuple(cfg.crops.rgb_mean)
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), tuple(round(m * 255) for m in mean))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    tensor = pil_to_tensor(canvas).float().div_(255)
    image.close()
    canvas.close()
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(list(cfg.crops.rgb_std))[:, None, None]


@torch.inference_mode()
def forward(model, inputs):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = model(inputs.cuda(non_blocking=True), is_training=True)['x_norm_clstoken'].float()
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError('nonfinite_encoder_output')
    if bool((features.norm(dim=-1) < 1e-10).any()):
        raise RuntimeError('zero_encoder_output')
    return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()


def calibrate(model, size):
    profiles = []
    for batch in [64, 128, 256]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            inputs = torch.zeros(batch, 3, size, size, device='cuda')
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                tick = time.monotonic()
                features = forward(model, inputs)
                torch.cuda.synchronize()
                times.append(time.monotonic() - tick)
                del features
            profiles.append(dict(batch=batch, seconds=statistics.median(times[1:]),
                                 peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30))
            del inputs
        except torch.OutOfMemoryError:
            if 'inputs' in locals():
                del inputs
            gc.collect()
            torch.cuda.empty_cache()
            break
    safe = [p for p in profiles if p['peak_reserved_GiB'] < 72]
    chosen = max(safe, key=lambda p: p['batch'] / p['seconds'])['batch'] if safe else 16
    return chosen, profiles


def main(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    folder = ROOT / 'evaluation'
    plan = load(folder / 'input_plan.json')
    if plan.get('world_size') != args.slots or digest(folder / 'inputs.private.jsonl') != plan['inputs_sha256']:
        raise RuntimeError('evaluation_input_plan_mismatch')
    with (folder / 'inputs.private.jsonl').open() as source:
        records = [json.loads(line) for line in source]
    indices = [i for i in range(len(records)) if i % args.slots == args.slot]
    destination = folder / 'features' / f'slot_{args.slot:02d}'
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / 'indices.private.npy', np.asarray(indices, dtype=np.int64))
    for kind in ['original_dinov3', 'craniofacial']:
        marker = destination / (kind + '.json')
        completed = load(marker)
        target = destination / (kind + '.private.npy')
        if completed.get('state') == 'complete' and completed.get('inputs_sha256') == plan['inputs_sha256'] and target.is_file() and digest(target) == completed.get('features_sha256'):
            continue
        model, cfg = encoder(kind)
        batch, profiles = calibrate(model, cfg.crops.global_crops_size)
        temporary = destination / (kind + '.private.tmp.npy')
        features = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                            shape=(len(indices), model.embed_dim))
        position = 0
        while position < len(indices):
            chosen = indices[position:position + batch]
            if any(digest(records[i]['crop_path']) != records[i]['crop_sha256'] for i in chosen):
                raise RuntimeError('released_evaluation_image_changed')
            inputs = torch.stack([fit_image(records[i]['crop_path'], cfg) for i in chosen])
            try:
                output = forward(model, inputs)
            except torch.OutOfMemoryError:
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                if batch <= 1:
                    raise
                batch = max(1, batch // 2)
                continue
            features[position:position + len(chosen)] = output
            position += len(chosen)
            del inputs, output
            save(destination / 'progress.json', dict(state='extracting', model=kind,
                 completed=position, assigned=len(indices), batch=batch, updated_at_utc=now()))
        features.flush()
        del features
        temporary.replace(target)
        save(marker, dict(state='complete', inputs_sha256=plan['inputs_sha256'],
             features_sha256=digest(target), rows=len(indices), inference_batch=batch,
             synthetic_batch_profiles=profiles, completed_at_utc=now()))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save(destination / 'completion.json', dict(state='complete', inputs_sha256=plan['inputs_sha256'],
                                               completed_at_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--slots', type=int, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        save(ROOT / 'evaluation/features' / f'slot_{args.slot:02d}' / 'failure.json',
             dict(state='failed', error_type=type(exc).__name__,
                  out_of_memory=isinstance(exc, torch.OutOfMemoryError), updated_at_utc=now()))
        raise SystemExit(1)
