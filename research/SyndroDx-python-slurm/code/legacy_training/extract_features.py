#!/usr/bin/env python
"""Extract frozen DINOv2 features for every cropped face in labels.tsv.

Frozen features first, SSL adaptation only if the probe says it is needed.
Writes features.npy (float16, N x D) aligned row-for-row with labels.tsv.
"""
import os, sys, csv, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


class Faces(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert('RGB').resize(
                (self.size, self.size), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.
            return (x - self.mean) / self.std, 1
        except Exception:
            return torch.zeros(3, self.size, self.size), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(HERE, 'labels.tsv'))
    ap.add_argument('--out',    default=os.path.join(HERE, 'features.npy'))
    ap.add_argument('--model',  default='facebook/dinov2-large')
    ap.add_argument('--resized-manifest', default=os.path.join(HERE, 'resized_manifest.tsv'))
    ap.add_argument('--size',   type=int, default=224)
    ap.add_argument('--batch',  type=int, default=256)
    ap.add_argument('--workers',type=int, default=16)
    args = ap.parse_args()

    resized = {}
    if args.resized_manifest and os.path.exists(args.resized_manifest):
        with open(args.resized_manifest, newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                resized[r['img_id']] = r['resized_path']
    paths = []
    with open(args.labels, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            paths.append(resized.get(r['img_id'], r['cropping_path']))
    n_res = sum(1 for p in paths if p.startswith(os.path.dirname(HERE)) or 'faces_384' in p)
    print(f"using pre-resized copies for {n_res:,}/{len(paths):,} images", flush=True)
    print(f"{len(paths):,} images", flush=True)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    dim = model.config.hidden_size
    print(f"{args.model}  dim={dim}", flush=True)

    dl = DataLoader(Faces(paths, args.size), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)
    out = np.zeros((len(paths), dim), dtype=np.float16)
    ok  = np.zeros(len(paths), dtype=bool)
    i = 0
    with torch.inference_mode():
        for x, good in dl:
            x = x.cuda(non_blocking=True).half()
            h = model(pixel_values=x).last_hidden_state
            # CLS token concatenated with mean of patch tokens is the standard
            # DINOv2 evaluation feature; keep CLS only to halve storage.
            f = h[:, 0]
            n = f.shape[0]
            out[i:i+n] = f.float().cpu().numpy().astype(np.float16)
            ok[i:i+n] = good.numpy().astype(bool)
            i += n
            if i % (args.batch * 20) == 0:
                print(f"  {i:,}/{len(paths):,}", flush=True)
    np.save(args.out, out)
    np.save(args.out.replace('.npy', '_ok.npy'), ok)
    print(f"wrote {args.out}  shape={out.shape}  unreadable={int((~ok).sum())}")


if __name__ == '__main__':
    main()
