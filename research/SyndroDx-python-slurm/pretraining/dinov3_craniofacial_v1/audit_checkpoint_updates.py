"""Read-only local optimizer/parameter audit. Never prints tensor values."""
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from runtime import ROOT, save, now
from weights import OfficialBackbone


def small_state(path):
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    selected = {}
    for name, value in metadata.state_dict_metadata.items():
        if not hasattr(value, 'size'):
            continue
        suffixes = ['backbone.patch_embed.proj.weight', 'backbone.norm.weight', 'dino_head.last_layer.weight']
        model = name.startswith('model.') and any(name.endswith(suffix) for suffix in suffixes)
        optimizer = name.startswith('optimizer.') and 'student.backbone.patch_embed.proj.weight' in name and name.endswith(('exp_avg', 'exp_avg_sq', 'step'))
        if model or optimizer:
            selected[name] = torch.empty(tuple(value.size), dtype=value.properties.dtype)
    dcp.load(selected, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    return selected


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main():
    torch.set_num_threads(1)
    directory = ROOT / 'training/ckpt'
    checkpoints = sorted([p for p in directory.iterdir() if p.is_dir() and p.name.isdigit() and (p / '.metadata').is_file()], key=lambda p: int(p.name))
    if len(checkpoints) < 2:
        raise RuntimeError('two_committed_checkpoints_required')
    first, last = checkpoints[0], checkpoints[-1]
    old, current = small_state(first), small_state(last)
    official = OfficialBackbone(ROOT / 'assets/dinov3-vit7b16')
    public_patch = official.tensor('patch_embed.proj.weight')
    report = dict(state='passed', checkpoint_iterations=[int(first.name), int(last.name)],
                  parameter_change={}, optimizer={}, clinical_images_read=False, tensor_values_exported=False,
                  conclusions_limited_to='parameter updates; not evidence of useful representations', created_at_utc=now())
    for name, value in current.items():
        if name.startswith('optimizer.'):
            report['optimizer'][name] = dict(finite=bool(torch.isfinite(value).all()),
                                            nonzero=bool(torch.count_nonzero(value)), rms=float(value.float().square().mean().sqrt()))
        elif name in old:
            report['parameter_change'][name] = dict(finite=bool(torch.isfinite(value).all()), relative_l2_between_checkpoints=relative(value, old[name]))
            if name.endswith('backbone.patch_embed.proj.weight'):
                report['parameter_change'][name]['relative_l2_from_official'] = relative(value, public_patch)
    student, teacher = 'model.student.backbone.patch_embed.proj.weight', 'model.teacher.backbone.patch_embed.proj.weight'
    if student in current and teacher in current:
        report['teacher_student_patch_relative_l2'] = relative(current[teacher], current[student])
    report['encoder_updates_detected'] = any(v['relative_l2_between_checkpoints'] > 0 for k, v in report['parameter_change'].items() if '.backbone.' in k)
    report['optimizer_moments_nonzero'] = any(v['nonzero'] for k, v in report['optimizer'].items() if k.endswith('exp_avg'))
    save(ROOT / 'training/diagnostics/checkpoint_update_audit.json', report)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(ROOT / 'training/diagnostics/checkpoint_update_audit.json', dict(state='failed', error_type=type(exc).__name__, created_at_utc=now()))
        raise SystemExit(1)
