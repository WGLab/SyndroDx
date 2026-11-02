#!/usr/bin/env python
"""Has the DINO encoder collapsed?

The degenerate solution in self-distillation is for the student to match the
teacher regardless of input -- the teacher is only an EMA of the student, so
agreement is cheap and the loss falls fast while the representation carries no
information. Centering + sharpening are meant to prevent it; this measures
whether they did.

Three signals, compared against the frozen DINOv2 init on identical images:
  feature std           -> collapses toward 0
  effective rank        -> participation ratio of the covariance spectrum;
                           collapses toward 1 (all images on one direction)
  mean pairwise cosine  -> collapses toward 1 (every face looks alike)
"""
import argparse, os, csv, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
csv.field_size_limit(10**9)


def load_imgs(paths, size=224):
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert('RGB').resize((size, size), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.
            out.append((x - MEAN) / STD)
        except Exception:
            pass
    return torch.stack(out)


@torch.no_grad()
def feats(model, X, bs=128):
    o = []
    for i in range(0, len(X), bs):
        h = model(pixel_values=X[i:i+bs].cuda().half(),
                  interpolate_pos_encoding=True).last_hidden_state[:, 0]
        o.append(h.float().cpu())
    return torch.cat(o).numpy()


def stats(F, tag):
    Fc = F - F.mean(0, keepdims=True)
    C = np.cov(Fc, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    eff = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12)     # participation ratio
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    S = Fn @ Fn.T
    iu = np.triu_indices(len(S), 1)
    print(f"  {tag:<26} std={F.std():7.4f}  eff_rank={eff:7.2f}/{F.shape[1]}  "
          f"cos_mean={S[iu].mean():+.4f}  cos_p95={np.percentile(S[iu],95):+.4f}")
    return eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--base', default='facebook/dinov2-base')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(HERE, 'labels.tsv'), newline=''), delimiter='\t'))
    man = {r['img_id']: r['resized_path'] for r in
           csv.DictReader(open(os.path.join(HERE, 'resized_manifest.tsv'), newline=''), delimiter='\t')}
    rng = np.random.RandomState(0)
    sel = rng.choice(len(rows), args.n, replace=False)
    paths = [man.get(rows[i]['img_id'], rows[i]['cropping_path']) for i in sel]
    X = load_imgs(paths)
    print(f"{len(X)} images\n")

    from transformers import AutoModel
    ref = AutoModel.from_pretrained(args.base, torch_dtype=torch.float16).cuda().eval()
    e_ref = stats(feats(ref, X), 'frozen DINOv2-base')

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    for which in ('teacher_backbone', 'student_backbone'):
        m = AutoModel.from_pretrained(args.base, torch_dtype=torch.float16)
        m.load_state_dict({k: v.half() for k, v in ck[which].items()}, strict=True)
        e = stats(feats(m.cuda().eval(), X), f"ep{ck['epoch']} {which.split('_')[0]}")
        del m; torch.cuda.empty_cache()
        print(f"     effective-rank retained vs frozen: {100*e/e_ref:5.1f}%")
    print("\nVERDICT: collapse would show as eff_rank -> ~1 and cos_mean -> ~+1.")


if __name__ == '__main__':
    main()
