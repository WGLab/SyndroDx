#!/usr/bin/env python
"""DINO self-distillation, continued from DINOv2 weights, on craniofacial faces.

Why DINO rather than MAE or SimCLR:
  * Self-distillation needs no negatives, so we never have to decide whether two
    photos of the same patient are a positive or a negative -- with 7.4 photos
    per patient, a contrastive objective would spend its capacity pushing apart
    images of the same face.
  * It starts cleanly from DINOv2 weights (DINOv2 = DINO + iBOT + KoLeo), so this
    is genuine domain adaptation of a strong encoder rather than from-scratch
    training on 155k images, which is far too few.
  * Its features linear-probe well, which is how we measure whether the
    adaptation actually bought anything.

Run with torchrun; see run_ssl.sbatch.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ssl_dataset import CraniofacialSSL, collate


# --------------------------------------------------------------------------- #
class DinoHead(nn.Module):
    """3-layer MLP -> L2-normalised bottleneck -> weight-normed prototype layer."""

    def __init__(self, in_dim, out_dim=65536, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, out_dim, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        return self.last(nn.functional.normalize(x, dim=-1, p=2))


class Wrapper(nn.Module):
    """Backbone + head. Groups crops by resolution so each resolution is one
    forward pass rather than one per crop."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops):
        sizes = [c.shape[-1] for c in crops]
        out, idx = [], []
        for s in sorted(set(sizes)):
            sel = [i for i, x in enumerate(sizes) if x == s]
            x = torch.cat([crops[i] for i in sel])
            h = self.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            out.append(h)
            idx.extend(sel)
        h = torch.cat(out)
        # restore the original crop ordering
        order = np.argsort(np.asarray(idx), kind='stable')
        per = crops[0].shape[0]
        chunks = list(torch.split(h, per))
        h = torch.cat([chunks[i] for i in order])
        return self.head(h)


class DinoLoss(nn.Module):
    def __init__(self, out_dim, n_crops, warmup_teacher_temp, teacher_temp,
                 warmup_epochs, n_epochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp, self.cm, self.n_crops = student_temp, center_momentum, n_crops
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.temps = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(max(0, n_epochs - warmup_epochs)) * teacher_temp])

    @torch.no_grad()
    def diagnostics(self, t_out, epoch=0):
        """Standard DINO collapse monitors.

        per_sample_H : entropy of each teacher distribution. -> 0 means the
                       teacher is one-hot and the student can memorise it.
        batch_mean_H : entropy of the batch-AVERAGED teacher distribution. This
                       is the real collapse signal -- if every image maps to the
                       same prototype it falls toward 0 while per_sample_H also
                       falls. Healthy training keeps batch_mean_H high
                       (near ln(out_dim)) with per_sample_H lower.
        active       : fraction of prototypes receiving non-trivial mass.
        """
        t = self.temps[min(epoch, len(self.temps) - 1)]
        p = torch.softmax((t_out - self.center) / t, dim=-1)
        per = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
        m = p.mean(0)
        bm = -(m * torch.log(m.clamp_min(1e-12))).sum()
        active = (m > 0.1 / m.numel()).float().mean()   # >10% of uniform mass
        return float(per), float(bm), float(active), p.sum(0).detach()

    def forward(self, s_out, t_out, epoch):
        s = (s_out / self.student_temp).chunk(self.n_crops)
        t = torch.softmax((t_out - self.center) / self.temps[min(epoch, len(self.temps)-1)], dim=-1).detach().chunk(2)
        loss, n = 0.0, 0
        for ti, tq in enumerate(t):
            for si, sq in enumerate(s):
                if si == ti:            # never distil a view against itself
                    continue
                loss = loss + torch.sum(-tq * torch.log_softmax(sq, dim=-1), dim=-1).mean()
                n += 1
        self.update_center(t_out)
        return loss / n

    @torch.no_grad()
    def update_center(self, t_out):
        c = torch.sum(t_out, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(c)
            c = c / (len(t_out) * dist.get_world_size())
        else:
            c = c / len(t_out)
        self.center = self.center * self.cm + c * (1 - self.cm)


def cosine_sched(base, final, epochs, steps_per_epoch, warmup_epochs=0, start=0):
    w = np.linspace(start, base, warmup_epochs * steps_per_epoch) if warmup_epochs else np.array([])
    n = epochs * steps_per_epoch - len(w)
    it = np.arange(n)
    c = final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / n))
    return np.concatenate([w, c])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', default=os.path.join(HERE, 'labels.tsv'))
    p.add_argument('--out-dir', default=os.path.join(HERE, 'ssl_ckpt'))
    p.add_argument('--backbone', default='facebook/dinov2-base')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=48, help='per GPU')
    p.add_argument('--lr', type=float, default=5e-5, help='post-warmup peak, scaled by batch/256')
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--wd', type=float, default=0.04)
    p.add_argument('--wd-end', type=float, default=0.2)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--momentum', type=float, default=0.996)
    p.add_argument('--out-dim', type=int, default=65536)
    p.add_argument('--n-local', type=int, default=6)
    p.add_argument('--gsize', type=int, default=224)
    p.add_argument('--lsize', type=int, default=98)
    p.add_argument('--session-pair-prob', type=float, default=0.5)
    p.add_argument('--teacher-temp', type=float, default=0.04)
    p.add_argument('--warmup-teacher-temp', type=float, default=0.02)
    p.add_argument('--teacher-temp-warmup-epochs', type=int, default=20)
    p.add_argument('--freeze-last-layer', type=int, default=1)
    p.add_argument('--clip-grad', type=float, default=3.0)
    p.add_argument('--abort-proto-frac', type=float, default=0.02,
                   help='collapse guard: abort if effective prototypes per batch '
                        'fall below this fraction of the per-GPU batch size. '
                        'H_batchmean is capped at ln(batch) once the teacher '
                        'sharpens, so an absolute entropy floor is wrong.')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-every', type=int, default=2)
    p.add_argument('--resized-manifest',
                   default=os.path.join(HERE, 'resized_manifest.tsv'),
                   help='img_id -> pre-resized path; pass "" to read the originals')
    p.add_argument('--limit-steps', type=int, default=0, help='smoke test: stop after N steps')
    p.add_argument('--exclude-eval-splits', default='',
                   help="comma list, e.g. 'val,test' to keep eval patients out of pretraining")
    args = p.parse_args()

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    is_main = rank == 0
    torch.manual_seed(0); np.random.seed(0)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(json.dumps(vars(args), indent=2), flush=True)

    excl = [s for s in args.exclude_eval_splits.split(',') if s]
    ds = CraniofacialSSL(args.labels, gsize=args.gsize, lsize=args.lsize,
                         n_local=args.n_local, session_pair_prob=args.session_pair_prob,
                         exclude_split=excl or None,
                         resized_manifest=args.resized_manifest or None)
    sampler = DistributedSampler(ds, shuffle=True) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                    shuffle=(sampler is None), num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=collate,
                    persistent_workers=True, prefetch_factor=4)

    from transformers import AutoModel
    s_bb = AutoModel.from_pretrained(args.backbone)
    t_bb = AutoModel.from_pretrained(args.backbone)
    dim = s_bb.config.hidden_size
    student = Wrapper(s_bb, DinoHead(dim, args.out_dim)).to(dev)
    teacher = Wrapper(t_bb, DinoHead(dim, args.out_dim)).to(dev)
    teacher.load_state_dict(student.state_dict())
    for q in teacher.parameters():
        q.requires_grad = False
    # DINOv2 carries a mask_token used only for masked-image modelling. A plain
    # multi-crop forward never touches it, so it receives no gradient and DDP
    # aborts with "parameters that were not used in producing loss". Freezing it
    # is correct and avoids the cost of find_unused_parameters=True.
    n_frozen = 0
    for n, q in student.named_parameters():
        if 'mask_token' in n:
            q.requires_grad = False
            n_frozen += 1
    if is_main:
        print(f"[init] froze {n_frozen} unused mask_token parameter(s) for DDP", flush=True)

    if world > 1:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)
    raw = student.module if world > 1 else student

    n_crops = 2 + args.n_local
    crit = DinoLoss(args.out_dim, n_crops, args.warmup_teacher_temp, args.teacher_temp,
                    args.teacher_temp_warmup_epochs, args.epochs).to(dev)

    decay, no_decay = [], []
    for n, q in raw.named_parameters():
        if not q.requires_grad:
            continue
        (no_decay if q.ndim == 1 or n.endswith('.bias') else decay).append(q)
    opt = torch.optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0.}])

    spe = len(dl)
    eff_lr = args.lr * (args.batch_size * world) / 256.
    lr_s = cosine_sched(eff_lr, args.min_lr, args.epochs, spe, args.warmup_epochs)
    wd_s = cosine_sched(args.wd, args.wd_end, args.epochs, spe)
    mo_s = cosine_sched(args.momentum, 1.0, args.epochs, spe)
    if is_main:
        print(f"[train] {len(ds):,} imgs | {spe} steps/epoch | world={world} | "
              f"global batch={args.batch_size*world} | peak lr={eff_lr:.2e} | "
              f"{n_crops} crops/sample", flush=True)

    step = 0
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        t0, tot, seen = time.time(), 0.0, 0
        usage = torch.zeros(args.out_dim, device=dev)
        for crops in dl:
            for i, g in enumerate(opt.param_groups):
                g['lr'] = lr_s[step]
                if i == 0:
                    g['weight_decay'] = wd_s[step]
            crops = [c.to(dev, non_blocking=True) for c in crops]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                t_out = teacher(crops[:2])
                s_out = student(crops)
                loss = crit(s_out.float(), t_out.float(), epoch)
            if not torch.isfinite(loss):
                if is_main:
                    print(f"  !! non-finite loss at step {step}, skipping", flush=True)
                opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip_grad)
            if epoch < args.freeze_last_layer:
                for n, q in raw.named_parameters():
                    if 'last' in n:
                        q.grad = None
            opt.step()
            with torch.no_grad():
                m = mo_s[step]
                for ps, pt in zip(raw.parameters(), teacher.parameters()):
                    pt.mul_(m).add_((1 - m) * ps.detach())
            with torch.no_grad():
                _, _, _, up = crit.diagnostics(t_out.float(), epoch)
                usage += up
            tot += float(loss.detach()) * crops[0].shape[0]
            seen += crops[0].shape[0]
            step += 1
            if args.limit_steps and step >= args.limit_steps:
                if is_main:
                    print(f"[smoke] stopping at step {step}, loss={tot/seen:.4f}", flush=True)
                break
            if is_main and step % 50 == 0:
                ph, bh, act, _ = crit.diagnostics(t_out.float(), epoch)
                import math as _m
                eff = _m.exp(bh)
                floor = args.abort_proto_frac * args.batch_size
                if args.abort_proto_frac and step > 400 and eff < floor:
                    print(f"  !! ABORT: {eff:.0f} effective prototypes for a "
                          f"{args.batch_size}-image batch (floor {floor:.0f}) -- images "
                          f"are sharing prototypes, i.e. real collapse. "
                          f"H_sample={ph:.3f} H_batchmean={bh:.3f}", flush=True)
                    raise SystemExit(17)
                print(f"  ep{epoch+1} step{step} loss={tot/seen:.4f} "
                      f"H_sample={ph:.3f} H_batchmean={bh:.3f} eff_proto={eff:.0f}/{args.batch_size} "
                      f"active={act:.4f} "
                      f"lr={lr_s[step-1]:.2e} m={mo_s[step-1]:.4f} "
                      f"({(time.time()-t0)/max(seen,1)*1000:.1f} ms/img)", flush=True)
        if args.limit_steps and step >= args.limit_steps:
            if is_main:
                print("[smoke] done", flush=True)
            break
        if world > 1:
            dist.all_reduce(usage)
        u = usage / usage.sum().clamp_min(1e-12)
        H_ep = float(-(u * torch.log(u.clamp_min(1e-12))).sum())
        if is_main:
            import math as _m
            print(f"[epoch {epoch+1}] prototype usage over the full epoch: "
                  f"H={H_ep:.3f}/{_m.log(args.out_dim):.2f}  "
                  f"eff_proto={_m.exp(H_ep):.0f}/{args.out_dim}", flush=True)
        if is_main:
            print(f"[epoch {epoch+1}/{args.epochs}] loss={tot/max(seen,1):.4f} "
                  f"time={(time.time()-t0)/60:.1f} min", flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                ck = os.path.join(args.out_dir, f'ckpt_ep{epoch+1:03d}.pt')
                torch.save({'epoch': epoch + 1, 'args': vars(args),
                            'teacher_backbone': teacher.backbone.state_dict(),
                            'student_backbone': raw.backbone.state_dict()}, ck)
                # NOTE: deliberately NOT calling save_pretrained() here.
                # transformers' save_pretrained -> accelerate.unwrap_model ->
                # imports deepspeed -> deepspeed probes CUDA_HOME and raises
                # MissingCUDAException under Slurm, killing the run *after* a
                # perfectly good checkpoint was already written. The .pt above
                # holds the backbone state_dict; load it with
                # AutoModel.from_pretrained(base) + load_state_dict, which is
                # what check_collapse.py and the probe already do.
                print(f"  saved {ck}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
